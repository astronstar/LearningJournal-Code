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


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def load_pkl(file):
	with open(file, 'rb') as file:
		result = pickle.load(file)
		return result


def get_metrics(y_test, predictions):
	print('accuracy_score：', accuracy_score(y_test, predictions))
	print('precision_score：', precision_score(y_test, predictions))
	print('recall_score：', recall_score(y_test, predictions))
	print('f1_score：', f1_score(y_test, predictions))


def exponent_neg_manhattan_distance(s1, s2):
	return K.exp(-K.sum(K.abs(s1 - s2), axis=1, keepdims=True))


def get_arci_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, filter_1d_num, drop_rate):
	s1 = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
	s2 = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

	embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

	s1_embed = embedding_layer(s1)
	s2_embed = embedding_layer(s2)

	input_data = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_DIM))
	x_0 = Conv1D(filter_1d_num, kernel_size=1, padding='same', kernel_initializer='normal', activation='relu')(input_data)
	x_1 = Conv1D(filter_1d_num, kernel_size=2, padding='same', kernel_initializer='normal', activation='relu')(input_data)
	x_2 = Conv1D(filter_1d_num, kernel_size=3, padding='same', kernel_initializer='normal', activation='relu')(input_data)
	x_3 = Conv1D(filter_1d_num, kernel_size=4, padding='same', kernel_initializer='normal', activation='relu')(input_data)
	x_4 = Conv1D(filter_1d_num, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(input_data)

	maxpool_0 = MaxPool1D(pool_size=MAX_SENTENCE_LENGTH - 1 + 1)(x_0)
	maxpool_1 = MaxPool1D(pool_size=MAX_SENTENCE_LENGTH - 2 + 1)(x_1)
	maxpool_2 = MaxPool1D(pool_size=MAX_SENTENCE_LENGTH - 3 + 1)(x_2)
	maxpool_3 = MaxPool1D(pool_size=MAX_SENTENCE_LENGTH - 4 + 1)(x_3)
	maxpool_4 = MaxPool1D(pool_size=MAX_SENTENCE_LENGTH - 5 + 1)(x_4)

	x = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])

	shared_layer = Model(inputs=input_data, outputs=x)

	s1_pool = shared_layer(s1_embed)
	s2_pool = shared_layer(s2_embed)

	z = Concatenate(axis=1)([s1_pool, s2_pool])

	pool_flat = Flatten()(z)
	pool_flat_drop = Dropout(rate=drop_rate)(pool_flat)
	pool_norm = BatchNormalization()(pool_flat_drop)

	mlp = Dense(128, activation='relu')(pool_norm)
	mlp = Dense(64, activation='relu')(mlp)
	mlp = Dense(32, activation='relu')(mlp)

	out = Dense(1, activation='sigmoid')(mlp)

	model = Model(inputs=[s1, s2], outputs=out)

	loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]
	model.compile(optimizer=Adam(lr=0.001, beta_1=0.8), loss=loss, metrics=metrics)
	print(model.summary())

	return model


def arci_train():
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
	MAX_SENTENCE_LENGTH = 20
	# filter_1d_num = [256, 128, 64]
	filter_num = 256
	embedding_matrix = load_pkl('./data/model/train_word_w2v_embedding_matrix_skip.pkl')  # 词级 skip
	# drop_rates = [0.6, 0.4, 0.2, 0.05]
	drop_rate = 0.6

	patience = 8
	EPOCHS = 90
	train_batch_size = 64
	test_batch_size = 500

	filepath = './arci/' + 'arci'  + time.strftime("_%m-%d %H-%M-%S") + ".h5"   # 每次运行的模型都进行保存，不覆盖之前的结果
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto')
	callbacks = [checkpoint, earlystop]
	# for filter_num in filter_1d_num:
	# 	for drop_rate in drop_rates:
	# 		model = get_arci_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, filter_num, drop_rate)
	# 		model.fit(x=[s1_train, s2_train], y=y_train, validation_data=([s1_val, s2_val], y_val), batch_size=train_batch_size, 
	# 				callbacks=callbacks, epochs=EPOCHS, verbose=2)

	# 		y_pred = model.predict([s1_test, s2_test], batch_size=test_batch_size)

	# 		r1, r2, r3 = r_f1_thresh(y_pred, y_test)
	# 		print('filter_num:', filter_num, 'drop_rate:', drop_rate, 'R相关系数:', r1, '最优评分:', r2, 'f1阈值:', r3)

	# 		get_metrics(y_test, (y_pred >= r3))

	# 		K.clear_session()
	# 		tf.reset_default_graph()

	model = get_arci_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, filter_num, drop_rate)
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
	arci_train()