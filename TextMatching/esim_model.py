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
import keras.models
import tempfile


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def save_pkl(file, object):
	with open(file, 'wb') as file:
		pickle.dump(object, file)

def load_pkl(file):
	with open(file, 'rb') as file:
		result = pickle.load(file)
		return result


def get_metrics(y_test, predictions):
	print('accuracy_score：', accuracy_score(y_test, predictions))
	print('precision_score：', precision_score(y_test, predictions))
	print('recall_score：', recall_score(y_test, predictions))
	print('f1_score：', f1_score(y_test, predictions))

def unchange_shape(input_shape):
	return input_shape

def substract(input_1, input_2):
	neg_input_2 = Lambda(lambda x: -x, output_shape=unchange_shape)(input_2)
	out_ = Add()([input_1, neg_input_2])

	return out_

def submult(input_1, input_2):
	mult = Multiply()([input_1, input_2])
	sub = substract(input_1, input_2)
	out_ = Concatenate()([sub, mult])

	return out_

def apply_multiple(input_, layers):
	if not len(layers) > 1:
		raise ValueError('Layers list should contain more than 1 layers')
	else:
		agg_ = []
		for layer in layers:
			agg_.append(layer(input_))
		out_ = Concatenate()(agg_)

	return out_

def soft_attention_alignment(input_1, input_2):
	attention = Dot(axes=-1)([input_1, input_2])
	w_att_1 = Lambda(lambda x: softmax(x, axis=1), output_shape=unchange_shape)(attention)
	w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=unchange_shape)(attention))

	in1_aligned = Dot(axes=1)([w_att_1, input_1])
	in2_aligned = Dot(axes=1)([w_att_2, input_2])

	return in1_aligned, in2_aligned

def get_esim_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_size, drop_rate):
	s1 = Input(shape=(MAX_SENTENCE_LENGTH, ), dtype='int32')
	s2 = Input(shape=(MAX_SENTENCE_LENGTH, ), dtype='int32')

	embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)
	bn = BatchNormalization(axis=2)
	s1_embed = bn(embedding_layer(s1))
	s2_embed = bn(embedding_layer(s2))

	lstm_layer = Bidirectional(LSTM(lstm_size, return_sequences=True))
	s1_lstm = lstm_layer(s1_embed)
	s2_lstm = lstm_layer(s2_embed)

	s1_aligned, s2_aligned = soft_attention_alignment(s1_lstm, s2_lstm)

	s1_combined = Concatenate()([s1_lstm, s2_aligned, submult(s1_lstm, s2_aligned)])
	s2_combined = Concatenate()([s2_lstm, s1_aligned, submult(s2_lstm, s1_aligned)])

	compose = Bidirectional(LSTM(lstm_size, return_sequences=True))
	s1_compare = compose(s1_combined)
	s2_compare = compose(s2_combined)

	s1_rep = apply_multiple(s1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
	s2_rep = apply_multiple(s2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

	merged = Concatenate()([s1_rep, s2_rep])

	dense = BatchNormalization()(merged)
	dense = Dense(128, activation='elu')(dense)
	dense = BatchNormalization()(dense)
	dense = Dropout(drop_rate)(dense)
	dense = Dense(64, activation='elu')(dense)
	dense = BatchNormalization()(dense)
	dense = Dropout(drop_rate)(dense)
	out = Dense(1, activation='sigmoid')(dense)

	model = Model(inputs=[s1, s2], outputs=out)

	loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]
	model.compile(optimizer=Adam(lr=0.001, beta_1=0.8), loss=loss, metrics=metrics)
	print(model.summary())

	return model


def esim_train():
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
	lstm_sizes = 128
	embedding_matrix = load_pkl('./data/model/train_word_w2v_embedding_matrix_skip.pkl')  # 词级 skip
	# drop_rates = [0.6, 0.4, 0.2, 0.05]
	drop_rate = 0.6

	patience = 8
	EPOCHS = 90
	train_batch_size = 64
	test_batch_size = 500

	filepath = './esim/' + 'esim'  + time.strftime("_%m-%d %H-%M-%S") + ".h5"   # 每次运行的模型都进行保存，不覆盖之前的结果
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto')
	callbacks = [checkpoint, earlystop]

	# for drop_rate in drop_rates:
	# 	model = get_esim_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_sizes, drop_rate)
	# 	model.fit(x=[s1_train, s2_train], y=y_train, validation_data=([s1_val, s2_val], y_val), batch_size=train_batch_size, 
	# 			callbacks=callbacks, epochs=EPOCHS, verbose=2)

	# 	y_pred = model.predict([s1_test, s2_test], batch_size=test_batch_size)

	# 	r1, r2, r3 = r_f1_thresh(y_pred, y_test)
	# 	print('drop_rate:', drop_rate, 'R相关系数:', r1, '最优评分:', r2, 'f1阈值:', r3)

	# 	get_metrics(y_test, (y_pred >= r3))

	# 	K.clear_session()
	# 	tf.reset_default_graph()

	model = get_esim_model(embedding_matrix, EMBEDDING_DIM, MAX_SENTENCE_LENGTH, lstm_sizes, drop_rate)
	model.fit(x=[s1_train, s2_train], y=y_train, validation_data=([s1_val, s2_val], y_val), batch_size=train_batch_size, 
			callbacks=callbacks, epochs=EPOCHS, verbose=2)

	y_pred = model.predict([s1_test, s2_test], batch_size=test_batch_size)
	predictions = (y_pred >= 0.28)

	get_metrics(y_test, predictions)

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
	esim_train()