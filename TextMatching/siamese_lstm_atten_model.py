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
from keras.regularizers import L1L2, l2
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		self.match_vector = None
		super(AttentionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		super(AttentionLayer, self).build(input_shape)

	def call(self, inputs, **kwargs):
		encode_s1 = inputs[0]
		encode_s2 = inputs[1]
		sentence_differerce = encode_s1 - encode_s2
		sentece_product = encode_s1 * encode_s2
		self.match_vector = K.concatenate([encode_s1, sentence_differerce, sentece_product, encode_s2], 1)

		return self.match_vector

	def compute_output_shape(self, input_shape):
		return K.int_shape(self.match_vector)


def load_pkl(file):
	with open(file, 'rb') as file:
		result = pickle.load(file)
		return result


def get_metrics(y_test, predictions):
	print('accuracy_score：', accuracy_score(y_test, predictions))
	print('precision_score：', precision_score(y_test, predictions))
	print('recall_score：', recall_score(y_test, predictions))
	print('f1_score：', f1_score(y_test, predictions))


def get_siamese_lstm_atten_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_layers, filter_1d_num, filter_1d_size, dense_dim, drop_rate):
	# step 1 定义孪生网络的公共层
	input_data = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_DIM))
	x = Dropout(drop_rate, )(input_data)

	for hidden_size in lstm_layers:
		x = Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)

	x = Conv1D(filter_1d_num, kernel_size=filter_1d_size, padding="valid", kernel_initializer="he_uniform")(x)
	x_p1 = GlobalAveragePooling1D()(x)
	x_p2 = GlobalMaxPooling1D()(x)
	x = Concatenate()([x_p1, x_p2])

	share_model = Model(inputs=input_data, outputs=x)

	# step 2 模型是多输入的结构，定义两个句子的输入
	s1 = Input(shape=(MAX_SENTENCE_LENGTH, ), dtype='int32')
	s2 = Input(shape=(MAX_SENTENCE_LENGTH, ), dtype='int32')

	embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)

	s1_embed = embedding_layer(s1)
	s2_embed = embedding_layer(s2)

	# Step3定义两个输入合并后的模型层
	s1_net = share_model(s1_embed)
	s2_net = share_model(s2_embed)

	matching_layer = AttentionLayer()([s1_net, s2_net])

	merge_model = Dense(dense_dim, activation='relu')(matching_layer)
	merge_model = Dropout(drop_rate)(merge_model)
	merge_model = BatchNormalization()(merge_model)
	merge_model = Dense(dense_dim, activation='relu')(merge_model)
	merge_model = Dropout(drop_rate)(merge_model)
	merge_model = BatchNormalization()(merge_model)

	# Step4 定义输出层
	output = Dense(1, activation='sigmoid')(merge_model)

	model = Model(inputs=[s1, s2], outputs=output)

	loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]
	model.compile(optimizer=Adam(lr=0.001, beta_1=0.8), loss=loss, metrics=metrics)
	print(model.summary())

	return model


def siamese_lstm_atten_train():
	s1_train = load_pkl('./data/preprocessed/s1_train_ids_pad.pkl')  # 词级
	s2_train = load_pkl('./data/preprocessed/s2_train_ids_pad.pkl')  # 词级
	y_train = load_pkl('./data/preprocessed/y_train.pkl')

	s1_val = load_pkl('./data/preprocessed/s1_val_ids_pad.pkl')  # 词级
	s2_val = load_pkl('./data/preprocessed/s2_val_ids_pad.pkl')  # 词级
	y_val = load_pkl('./data/preprocessed/y_val.pkl')

	s1_test = load_pkl('./data/preprocessed/s1_test_ids_pad.pkl')  # 词级
	s2_test = load_pkl('./data/preprocessed/s2_test_ids_pad.pkl')  # 词级
	y_test = load_pkl('./data/preprocessed/y_test.pkl')

	MAX_SENTENCE_LENGTH = 20
	EMBEDDING_DIM = 300
	lstm_layers = [128, 128, 128]
	filter_1d_num = [256, 128, 64]
	filter_1d_size = 2
	dense_dims = [256, 128, 64]
	embedding_matrix = load_pkl('./data/model/train_word_w2v_embedding_matrix_skip.pkl')  # 词级 skip
	drop_rates = [0.6, 0.4, 0.2, 0.05]

	patience = 8
	EPOCHS = 90
	train_batch_size = 64
	test_batch_size = 500

	filepath = './siamese_lstm_atten/' + 'siamese_lstm_atten'  + time.strftime("_%m-%d %H-%M-%S") + ".h5"   # 每次运行的模型都进行保存，不覆盖之前的结果
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto')
	callbacks = [checkpoint, earlystop]
	
	for filter_num in filter_1d_num:
		for dense_dim in dense_dims:
			for drop_rate in drop_rates:
				model = get_siamese_lstm_atten_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_layers, filter_num, 
													filter_1d_size, dense_dim, drop_rate)
				model.fit(x=[s1_train, s2_train], y=y_train, validation_data=([s1_val, s2_val], y_val), batch_size=train_batch_size, 
						callbacks=callbacks, epochs=EPOCHS, verbose=2)

				y_pred = model.predict([s1_test, s2_test], batch_size=test_batch_size)

				r1, r2, r3 = r_f1_thresh(y_pred, y_test)
				print('filter_num:', filter_num, 'dense_dim:', dense_dim, 'drop_rate:', drop_rate, 'R相关系数:', r1, '最优评分:', r2, 'f1阈值:', r3)

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
	siamese_lstm_atten_train()