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


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class AttentionLayer(Layer):
	def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None,
	             bias=True, **kwargs):
		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.step_dim = step_dim
		self.features_dim = 0

		super(AttentionLayer, self).__init__(**kwargs)

	def compute_mask(self, inputs, mask=None):
		return None

	def build(self, input_shape):
		assert len(input_shape) == 3
		self.W = self.add_weight((input_shape[-1],), initializer=self.init, name='{}_W'.format(self.name), regularizer=self.W_regularizer, constraint=self.W_constraint)
		self.features_dim = input_shape[-1]
		if self.bias:
			self.b = self.add_weight((input_shape[1],), initializer='zero', name='{}_b'.format(self.name), regularizer=self.b_regularizer, constraint=self.b_constraint)
		else:
			self.b = None
		self.built = True

	def call(self, x, mask=None):
		features_dim = self.features_dim
		step_dim = self.step_dim

		eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a

		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.features_dim

	def get_config(self):
		config = {'step_dim': self.step_dim}
		base_config = super(AttentionLayer, self).get_config()

		return dict(list(base_config.items()) + list(config.items()))

def load_pkl(file):
	with open(file, 'rb') as file:
		result = pickle.load(file)
		return result


def get_metrics(y_test, predictions):
	print('accuracy_score：', accuracy_score(y_test, predictions))
	print('precision_score：', precision_score(y_test, predictions))
	print('recall_score：', recall_score(y_test, predictions))
	print('f1_score：', f1_score(y_test, predictions))


def get_siamese_lstm_dssm_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_layers_num, filter_1d_num, filter_1d_size, conv2d_layers_num, filter_2d, filter_2d_size, pool_2d_size, drop_rate):
	s1_input = Input(shape=(MAX_SENTENCE_LENGTH, ), dtype='int32')
	s2_input = Input(shape=(MAX_SENTENCE_LENGTH, ), dtype='int32')

	embedding_layer1 = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

	att_layer1 = AttentionLayer(MAX_SENTENCE_LENGTH)
	bi_lstm_layer = Bidirectional(LSTM(lstm_layers_num))
	lstm_layer1 = LSTM(lstm_layers_num, return_sequences=True)
	lstm_layer2 = LSTM(lstm_layers_num)

	s1_embed = embedding_layer1(s1_input)
	s2_embed = embedding_layer1(s2_input)

	s1_lstm_lstm = lstm_layer2(lstm_layer1(s1_embed))
	s2_lstm_lstm = lstm_layer2(lstm_layer1(s2_embed))

	s1_lstm = lstm_layer1(s1_embed)
	s2_lstm = lstm_layer1(s2_embed)

	s1_bi = bi_lstm_layer(s1_embed)
	s2_bi = bi_lstm_layer(s2_embed)

	cnn_input_layer = dot([s1_lstm, s2_lstm], axes=-1)
	cnn_input_layer_dot = Reshape((MAX_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH, -1))(cnn_input_layer)

	layer_conv1 = Conv2D(filters=filter_1d_num, kernel_size=filter_1d_size, padding='same', activation='relu')(cnn_input_layer_dot)
	z = MaxPooling2D(pool_size=(2, 2))(layer_conv1)

	for i in range(conv2d_layers_num):
		z = Conv2D(filters=filter_2d[i], kernel_size=filter_2d_size[i], padding='same', activation='relu')(z)
		z = MaxPooling2D(pool_size=(pool_2d_size[i][0], pool_2d_size[i][1]))(z)

	pool1_flat = Flatten()(z)
	pool1_flat_drop = Dropout(drop_rate)(pool1_flat)

	ccn1 = Dense(32, activation='relu')(pool1_flat_drop)
	ccn2 = Dense(16, activation='relu')(ccn1)

	s1_att = att_layer1(s1_embed)
	s2_att = att_layer1(s2_embed)

	s1_last = Concatenate(axis=1)([s1_att, s1_bi])
	s2_last = Concatenate(axis=1)([s2_att, s2_bi])

	s1_s2_mul = Multiply()([s1_last, s2_last])
	s1_s2_sub = Lambda(lambda x: K.abs(x))(Subtract()([s1_last, s2_last]))
	s1_s2_maximum = Maximum()([Multiply()([s1_last, s1_last]), Multiply()([s2_last, s2_last])])
	s1_s2_sub1 = Lambda(lambda x: K.abs(x))(Subtract()([s1_lstm_lstm, s2_lstm_lstm]))

	last_layer = Concatenate(axis=1)([s1_s2_mul, s1_s2_sub, s1_s2_sub1, s1_s2_maximum])
	last_layer = Dropout(drop_rate)(last_layer)

	dense_layer1 = Dense(32, activation='relu')(last_layer)
	dense_layer2 = Dense(48, activation='relu')(last_layer)

	output_layer = Concatenate(axis=1)([dense_layer1, dense_layer2, s1_last, s2_last, ccn2])
	output_layer = Dense(1, activation='sigmoid')(output_layer)

	model = Model(inputs=[s1_input, s2_input], outputs=output_layer)

	loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]
	model.compile(optimizer=Adam(lr=0.001, beta_1=0.8), loss=loss, metrics=metrics)
	print(model.summary())

	return model

def siamese_lstm_dssm_train():
	s1_train = load_pkl('./data/preprocessed/s1_train_ids_pad.pkl')  # 词级
	s2_train = load_pkl('./data/preprocessed/s2_train_ids_pad.pkl')  # 词级
	y_train = load_pkl('./data/preprocessed/y_train.pkl')

	s1_val = load_pkl('./data/preprocessed/s1_val_ids_pad.pkl')  # 词级
	s2_val = load_pkl('./data/preprocessed/s2_val_ids_pad.pkl')  # 词级
	y_val = load_pkl('./data/preprocessed/y_val.pkl')

	s1_test = load_pkl('./data/preprocessed/s1_test_ids_pad.pkl')  # 词级
	s2_test = load_pkl('./data/preprocessed/s2_test_ids_pad.pkl')  # 词级
	y_test = load_pkl('./data/preprocessed/y_test.pkl')

	EMBEDDING_DIM = 300
	MAX_SENTENCE_LENGTH = 20  # 词级
	filter_1d_num = [256, 128, 64]
	filter_1d_size = 2
	lstm_layers_num = 256
	conv2d_layers_num = 2
	filter_2d = [256, 128]
	filter_2d_size = [[2, 2], [2, 2]]
	pool_2d_size = [[2, 2], [2, 2]]
	embedding_matrix = load_pkl('./data/model/train_word_w2v_embedding_matrix_skip.pkl')  # 词级 skip
	drop_rates = [0.6, 0.4, 0.2, 0.05]

	patience = 8
	EPOCHS = 90
	train_batch_size = 64
	test_batch_size = 500

	filepath = './siamese_lstm_dssm/' + 'siamese_lstm_dssm'  + time.strftime("_%m-%d %H-%M-%S") + ".h5"   # 每次运行的模型都进行保存，不覆盖之前的结果
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto')
	callbacks = [checkpoint, earlystop]
	for filter_num in filter_1d_num:
		for drop_rate in drop_rates:
			model = get_siamese_lstm_dssm_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_layers_num, filter_num, filter_1d_size, 
												conv2d_layers_num, filter_2d, filter_2d_size, pool_2d_size, drop_rate)
			model.fit(x=[s1_train, s2_train], y=y_train, validation_data=([s1_val, s2_val], y_val), batch_size=train_batch_size, 
					callbacks=callbacks, epochs=EPOCHS, verbose=2)

			y_pred = model.predict([s1_test, s2_test], batch_size=test_batch_size)

			r1, r2, r3 = r_f1_thresh(y_pred, y_test)
			print('filter_num:', filter_num, 'drop_rate:', drop_rate, 'R相关系数:', r1, '最优评分:', r2, 'f1阈值:', r3)

			get_metrics(y_test, (y_pred >= r3))

			K.clear_session()
			tf.reset_default_graph()


def r_f1_thresh(y_pred, y_true, step=1000):
	e = np.zeros((len(y_true), 2))

	e[:, 0] = y_pred.reshape(1, -1)
	e[:, 1] = y_true.reshape(1, -1)

	f = pd.DataFrame(e)
	thrs = np.linspace(0, 1, step+1)
	x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:,1]) for thr in thrs])
	f1_, thresh = max(x), thrs[x.argmax()]

	return f.corr()[0][1], f1_, thresh


if __name__ == '__main__':
	siamese_lstm_dssm_train()