# coding:utf-8

import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class Attention(Layer):
	def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None,
	             bias=True, **kwargs):
		"""
		Keras Layer that implements an Attention mechanism for temporal data.
		Supports Masking.
		Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
		# Input shape
			3D tensor with shape: `(samples, steps, features)`.
		# Output shape
			2D tensor with shape: `(samples, features)`.
		:param kwargs:
		Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
		The dimensions are inferred based on the output shape of the RNN.
		Example:
			model.add(LSTM(64, return_sequences=True))
			model.add(Attention())
		"""
		self.supports_masking = True
		# self.init = initializations.get('glorot_uniform')
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.step_dim = step_dim
		self.features_dim = 0
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='%s_W' % self.name,
		                         regularizer=self.W_regularizer, constraint=self.W_constraint)
		self.features_dim = input_shape[-1]

		if self.bias:
			self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='%s_b' % self.name,
			                         regularizer=self.b_regularizer, constraint=self.b_constraint)
		else:
			self.b = None

		self.built = True

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		# eij = K.dot(x, self.W) TF backend doesn't support it

		# features_dim = self.W.shape[0]
		# step_dim = x._keras_shape[1]

		features_dim = self.features_dim
		step_dim = self.step_dim
		eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)
		a = K.exp(eij)
		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			a *= K.cast(mask, K.floatx())

		# in some cases especially in the early stages of training the sum may be almost zero
		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a
		# print weigthted_input.shape
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		# return input_shape[0], input_shape[-1]
		return input_shape[0], self.features_dim


def load_pkl(file):
	with open(file, 'rb') as file:
		result = pickle.load(file)
		return result


def get_metrics(y_test, predictions):
	print('accuracy_score：', accuracy_score(y_test, predictions))
	print('precision_score：', precision_score(y_test, predictions))
	print('recall_score：', recall_score(y_test, predictions))
	print('f1_score：', f1_score(y_test, predictions))


def get_dssm_model(embedding_matrix_word, embedding_matrix_char, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH,
                   lstm_size, drop_rate):
	s1_word = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
	s2_word = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

	lstm0 = LSTM(lstm_size, return_sequences=True)
	lstm1 = Bidirectional(LSTM(lstm_size))
	lstm2 = LSTM(lstm_size)
	att1 = Attention(MAX_SENTENCE_LENGTH, name='att1')

	embedding_layer_word = Embedding(len(embedding_matrix_word), EMBEDDING_DIM, weights=[embedding_matrix_word],
	                                 trainable=False)
	embedding_layer_char = Embedding(len(embedding_matrix_char), EMBEDDING_DIM, weights=[embedding_matrix_char],
	                                 trainable=False)

	v1 = embedding_layer_word(s1_word)
	v2 = embedding_layer_word(s2_word)
	v11 = lstm1(v1)
	v22 = lstm1(v2)
	v11s = lstm2(lstm0(v1))
	v21s = lstm2(lstm0(v2))
	v1 = Concatenate(axis=1)([att1(v1), v11])
	v2 = Concatenate(axis=1)([att1(v2), v22])

	s1_char = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
	s2_char = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
	lstm1c = Bidirectional(LSTM(lstm_size))
	att1c = Attention(MAX_WORD_LENGTH, name='att1c')

	v1c = embedding_layer_char(s1_char)
	v2c = embedding_layer_char(s2_char)
	v11c = lstm1c(v1c)
	v22c = lstm1c(v2c)
	v1c = Concatenate(axis=1)([att1c(v1c), v11c])
	v2c = Concatenate(axis=1)([att1c(v2c), v22c])

	mul = Multiply()([v1, v2])
	sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
	maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])

	mulc = Multiply()([v1c, v2c])
	subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
	maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])

	sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v11s, v21s]))
	matchlist = Concatenate(axis=1)([mul, sub, mulc, subc, maximum, maximumc, sub2])
	matchlist = Dropout(drop_rate)(matchlist)

	matchlist = Concatenate(axis=1)(
		[Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])
	out = Dense(1, activation='sigmoid')(matchlist)

	model = Model(inputs=[s1_word, s2_word, s1_char, s2_char], outputs=out)

	loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]
	model.compile(optimizer=Adam(lr=0.001, beta_1=0.8), loss=loss, metrics=metrics)
	print(model.summary())

	return model


def dssm_train():
	s1_train_word = load_pkl('./data/preprocessed/s1_train_ids_pad.pkl')  # 词级
	s2_train_word = load_pkl('./data/preprocessed/s2_train_ids_pad.pkl')  # 词级
	s1_train_char = load_pkl('./data/preprocessed/s1_train_char_ids_pad.pkl')  # 字符级
	s2_train_char = load_pkl('./data/preprocessed/s2_train_char_ids_pad.pkl')  # 字符级

	y_train = load_pkl('./data/preprocessed/y_train.pkl')

	s1_val_word = load_pkl('./data/preprocessed/s1_val_ids_pad.pkl')  # 词级
	s2_val_word = load_pkl('./data/preprocessed/s2_val_ids_pad.pkl')  # 词级
	s1_val_char = load_pkl('./data/preprocessed/s1_val_char_ids_pad.pkl')  # 字符级
	s2_val_char = load_pkl('./data/preprocessed/s2_val_char_ids_pad.pkl')  # 字符级

	y_val = load_pkl('./data/preprocessed/y_val.pkl')

	s1_test_word = load_pkl('./data/preprocessed/s1_test_ids_pad.pkl')  # 词级
	s2_test_word = load_pkl('./data/preprocessed/s2_test_ids_pad.pkl')  # 词级
	s1_test_char = load_pkl('./data/preprocessed/s1_test_char_ids_pad.pkl')  # 词级
	s2_test_char = load_pkl('./data/preprocessed/s2_test_char_ids_pad.pkl')  # 词级

	y_test = load_pkl('./data/preprocessed/y_test.pkl')

	EMBEDDING_DIM = 300
	MAX_SENTENCE_LENGTH = 20
	MAX_WORD_LENGTH = 24
	lstm_size = 128

	embedding_matrix_word = load_pkl('./data/model/train_word_w2v_embedding_matrix_skip.pkl')  # 词级
	embedding_matrix_char = load_pkl('./data/model/train_char_w2v_embedding_matrix_skip.pkl')  # 字符级
	# drop_rates = [0.6, 0.4, 0.2, 0.05]
	drop_rate = 0.6

	patience = 8
	EPOCHS = 90
	train_batch_size = 64
	test_batch_size = 500

	filepath = './dssm/' + 'dssm' + time.strftime("_%m-%d %H-%M-%S") + ".h5"  # 每次运行的模型都进行保存，不覆盖之前的结果
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
	                             mode='auto')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto')
	callbacks = [checkpoint, earlystop]

	model = get_dssm_model(embedding_matrix_word, embedding_matrix_char, EMBEDDING_DIM, MAX_SENTENCE_LENGTH,
	                       MAX_WORD_LENGTH, lstm_size, drop_rate)
	model.fit(x=[s1_train_word, s2_train_word, s1_train_char, s2_train_char], y=y_train,
	          validation_data=([s1_val_word, s2_val_word, s1_val_char, s2_val_char], y_val),
	          batch_size=train_batch_size, callbacks=callbacks, epochs=EPOCHS, verbose=2)

	json_string = model.to_json()
	open('./dssm/dssm_model.json', 'w').write(json_string)
	model.save_weights('./dssm/dssm_weights.h5')

	y_pred = model.predict([s1_test_word, s2_test_word, s1_test_char, s2_test_char], batch_size=test_batch_size)

	r1, r2, r3 = r_f1_thresh(y_pred, y_test)
	print('drop_rate:', drop_rate, 'R相关系数:', r1, '最优评分:', r2, 'f1阈值:', r3)

	get_metrics(y_test, (y_pred >= r3))


# for drop_rate in drop_rates:
# 	model = get_dssm_model(embedding_matrix_word, embedding_matrix_char, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, MAX_WORD_LENGTH, lstm_size, drop_rate)
# 	model.fit(x=[s1_train_word, s2_train_word, s1_train_char, s2_train_char], y=y_train,
# 		validation_data=([s1_val_word, s2_val_word, s1_val_char, s2_val_char], y_val),
# 		batch_size=train_batch_size, callbacks=callbacks, epochs=EPOCHS, verbose=2)

# 	y_pred = model.predict([s1_test_word, s2_test_word, s1_test_char, s2_test_char], batch_size=test_batch_size)

# 	r1, r2, r3 = r_f1_thresh(y_pred, y_test)
# 	print('drop_rate:', drop_rate, 'R相关系数:', r1, '最优评分:', r2, 'f1阈值:', r3)

# 	get_metrics(y_test, (y_pred >= r3))

# 	K.clear_session()
# 	tf.reset_default_graph()


def r_f1_thresh(y_pred, y_true, step=1000):
	e = np.zeros((len(y_true), 2))

	e[:, 0] = y_pred.reshape(1, -1)
	e[:, 1] = y_true.reshape(1, -1)

	f = pd.DataFrame(e)
	thrs = np.linspace(0, 1, step + 1)
	x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:, 1]) for thr in thrs])
	f1_, thresh = max(x), thrs[x.argmax()]

	return f.corr()[0][1], f1_, thresh


if __name__ == '__main__':
	dssm_train()
