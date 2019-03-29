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


def get_arcii_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, filter_1d_num, conv2d_layers_num, filter_2d,
                    filter_2d_size, pool_2d_size, drop_rate):
	s1 = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
	s2 = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

	embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)

	s1_embed = embedding_layer(s1)
	s2_embed = embedding_layer(s2)

	input_data = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_DIM))
	x_0 = Conv1D(filter_1d_num, kernel_size=1, padding='same', kernel_initializer='normal', activation='relu')(
		input_data)
	x_1 = Conv1D(filter_1d_num, kernel_size=2, padding='same', kernel_initializer='normal', activation='relu')(
		input_data)
	x_2 = Conv1D(filter_1d_num, kernel_size=3, padding='same', kernel_initializer='normal', activation='relu')(
		input_data)
	x_3 = Conv1D(filter_1d_num, kernel_size=4, padding='same', kernel_initializer='normal', activation='relu')(
		input_data)
	x_4 = Conv1D(filter_1d_num, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(
		input_data)

	x = Concatenate(axis=1)([x_0, x_1, x_2, x_3, x_4])

	shared_conv1_layer = Model(inputs=input_data, outputs=x)

	s1_conv1 = shared_conv1_layer(s1_embed)
	s2_conv1 = shared_conv1_layer(s2_embed)

	s1_s2_conv1 = dot([s1_conv1, s2_conv1], axes=-1)

	s1_s2_conv1 = Reshape((MAX_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH, -1))(s1_s2_conv1)
	z = MaxPooling2D(pool_size=(2, 2))(s1_s2_conv1)

	for i in range(conv2d_layers_num):
		z = Conv2D(filters=filter_2d[i], kernel_size=filter_2d_size[i], padding='same', activation='relu')(z)
		z = MaxPooling2D(pool_size=(pool_2d_size[i][0], pool_2d_size[i][1]))(z)

	pool_flat = Flatten()(z)
	pool_flat_drop = Dropout(rate=drop_rate)(pool_flat)
	pool_norm = BatchNormalization()(pool_flat_drop)

	mlp = Dense(128)(pool_norm)
	mlp = Activation('relu')(mlp)
	mlp = Dense(64)(pool_norm)
	mlp = Activation('relu')(mlp)
	mlp = Dense(32)(pool_norm)
	mlp = Activation('relu')(mlp)

	out = Dense(1, activation='sigmoid')(mlp)

	model = Model(inputs=[s1, s2], outputs=out)

	loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]
	model.compile(optimizer=Adam(lr=0.001, beta_1=0.8), loss=loss, metrics=metrics)
	print(model.summary())

	return model


def arcii_train():
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
	filter_1d_num = [256, 128, 64]
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

	filepath = './arcii/' + 'arcii' + time.strftime("_%m-%d %H-%M-%S") + ".h5"  # 每次运行的模型都进行保存，不覆盖之前的结果

	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
	                             mode='auto')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto')
	callbacks = [checkpoint, earlystop]

	for filter_num in filter_1d_num:
		for drop_rate in drop_rates:
			model = get_arcii_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, filter_num, conv2d_layers_num,
			                        filter_2d, filter_2d_size, pool_2d_size, drop_rate)
			model.fit(x=[s1_train, s2_train], y=y_train, validation_data=([s1_val, s2_val], y_val),
			          batch_size=train_batch_size,
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
	thrs = np.linspace(0, 1, step + 1)
	x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:, 1]) for thr in thrs])
	f1_, thresh = max(x), thrs[x.argmax()]

	return f.corr()[0][1], f1_, thresh


if __name__ == '__main__':
	arcii_train()
