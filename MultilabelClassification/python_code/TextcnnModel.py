# coding:utf-8

"""
	作者：	仝哲
	日期：	2018年10月14日
	文件名：	TextcnnModel.py
	功能：	多标签分类——深度学习TextCNN模型训练及保存
"""

import tempfile
import jieba
import pickle
import pandas as pd
import numpy as np
import keras.models
from keras.models import Model
from keras.models import load_model
from keras.layers import Embedding, Input, Dense, SpatialDropout1D, concatenate, Conv1D, MaxPool1D
from keras.layers import Reshape, Flatten, Concatenate, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.callbacks import Callback
from gensim.models import word2vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_similarity_score, precision_score, recall_score, f1_score, accuracy_score, \
	hamming_loss
import warnings
import os
from DataLoad import Data_Load
from MyLog import Logger

np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ""


class JaccardEvaluation(Callback):
	"""
	自定义Keras中的回调函数
	"""

	def __init__(self, validation_data=(), interval=1):
		super(Callback, self).__init__()

		self.interval = interval
		self.X_val, self.y_val = validation_data
		self.accs = []
		self.highest = 0

	def on_epoch_end(self, epoch, logs={}):
		"""
		当本次迭代训练得分大于或等于前一次训练结果得分，就保存一次模型
		:param epoch:
		:param logs:
		:return:
		"""
		# print('accs', self.accs)
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.X_val, verbose=0, batch_size=128)
			y_pred = (y_pred > 0.5)
			score = jaccard_similarity_score(self.y_val, y_pred)
			self.accs.append(score)

			if score >= self.highest:
				self.highest = score
				self.model.save('./model/best_textcnn_model.h5')

			# print('\n jaccard - epoch: %d - score: %.6f \n' % (epoch + 1, score))

def make_keras_picklabel():
	"""
	依赖函数：pickle函数操作keras生成的.h5格式模型文件
	:return:
	"""

	def __getstate__(self):
		model_str = ""
		with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
			keras.models.save_model(self, fd.name, overwrite=True)
			model_str = fd.read()
		d = {'model_str': model_str}

		return d

	def __setstate__(self, state):
		with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
			fd.write(state['model_str'])
			fd.flush()
			model = keras.models.load_model(fd.name)

		self.__dict__ = model.__dict__

	cls = keras.models.Model
	cls.__getstate__ = __getstate__
	cls.__setstate__ = __setstate__

def train_data_cut(data):
	"""
	数据预处理，标签切割及语句分词
	:param data: 原始数据DataFrame格式，列名为 id, context, label
	:return: 处理后的data
	"""
	data['label'] = data['label'].apply(lambda x: x.split(','))
	data['context'] = data['context'].apply(lambda x: ' '.join(jieba.cut(x)))

	return data

def predict_data_cut(data):
	"""
	数据预处理，标签切割及语句分词
	:param data: 原始数据DataFrame格式，列名为 id, context, label
	:return: 处理后的data
	"""
	# data['label'] = data['label'].apply(lambda x: x.split(','))
	data['context'] = data['context'].apply(lambda x: ' '.join(jieba.cut(x)))

	return data

def get_label_model(data):
	"""
	获得标签二元化处理模型
	:param data: 标签数据
	:return: 标签二元化处理后的模型
	"""
	try:
		mlb = MultiLabelBinarizer()
		mlb.fit(list(data))

		return mlb
	except Exception as e:
		print('waring----label转换----')

	return None

def get_word_index(MAX_NB_WORDS, data):
	"""
	对词进行编码处理，即词-序号
	:param MAX_NB_WORDS: 处理的最大词量
	:param data: 待处理语句
	:return:
	"""
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(data)

	return tokenizer

def get_word2vec_model(data, w2v_size, w2v_window, w2v_min_count, w2v_negative):
	"""
	计算每个词的词向量
	:param data: 待处理语句
	:return: 词向量模型bin文件
	"""
	sentences = data.apply(lambda x: list(x.split(' ')))
	# sentences = data.apply(lambda x: list(jieba.cut(x)))

	model = word2vec.Word2Vec(sentences, sg=1, size=int(w2v_size), window=int(w2v_window), min_count=int(w2v_min_count),
							  negative=int(w2v_negative), sample=0.001, hs=1, workers=1)

	model.save('./w2v/word2vec_skip_mc.model')

	model.wv.save_word2vec_format('./w2v/word2vec_skip_mc.bin', binary=False)

	word_vectors = KeyedVectors.load_word2vec_format('./w2v/word2vec_skip_mc.bin', binary=False)

	return word_vectors

def get_word_vector(word_vectors):
	"""
	生成每个词的词向量
	:param word_vectors:
	:return:
	"""
	word_vectors = KeyedVectors.load_word2vec_format('./w2v/word2vec_skip_mc.bin', binary=False)
	print('有词向量的词数', len(list(word_vectors.wv.vocab)))

	embeddings_index = {}
	for i in word_vectors.wv.vocab:
		embedding = np.asarray(word_vectors[i], dtype='float')
		embeddings_index[i] = embedding
	print('word embedding', len(embeddings_index))

	return embeddings_index

def get_word_embedding_matrix(nb_words, EMBEDDING_DIM, word_index, MAX_NB_WORDS, embeddings_index):
	"""
	得到word_embedding_matrix
	:param nb_words:
	:param EMBEDDING_DIM:
	:param word_index:
	:param MAX_NB_WORDS:
	:param embeddings_index:
	:return:
	"""
	word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		if i > MAX_NB_WORDS:
			continue

		embedding_vector = embeddings_index.get(str(word).upper())

		if embedding_vector is not None:
			word_embedding_matrix[i] = embedding_vector

	return word_embedding_matrix

def get_textcnn_model(max_sequence_length, nb_words, EMBEDDING_DIM, word_embedding_matrix, drop_rate, num_filter, mlb):
	"""
	textCNN模型训练
	:param max_sequence_length:
	:param nb_words:
	:param EMBEDDING_DIM:
	:param word_embedding_matrix:
	:param drop_rate:
	:param num_filter:
	:param mlb:
	:return:
	"""
	filter_sizes = [1, 2, 3, 4, 5]

	inp = Input(shape=(max_sequence_length,))
	x = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix])(inp)
	x = SpatialDropout1D(drop_rate)(x)

	conv_0 = Conv1D(num_filter, kernel_size=filter_sizes[0], padding='same', kernel_initializer='normal',
					activation='elu')(x)
	conv_1 = Conv1D(num_filter, kernel_size=filter_sizes[1], padding='same', kernel_initializer='normal',
					activation='elu')(x)
	conv_2 = Conv1D(num_filter, kernel_size=filter_sizes[2], padding='same', kernel_initializer='normal',
					activation='elu')(x)
	conv_3 = Conv1D(num_filter, kernel_size=filter_sizes[3], padding='same', kernel_initializer='normal',
					activation='elu')(x)
	conv_4 = Conv1D(num_filter, kernel_size=filter_sizes[4], padding='same', kernel_initializer='normal',
					activation='elu')(x)

	maxpool_0 = MaxPool1D(pool_size=max_sequence_length - filter_sizes[0] + 1)(conv_0)
	maxpool_1 = MaxPool1D(pool_size=max_sequence_length - filter_sizes[1] + 1)(conv_1)
	maxpool_2 = MaxPool1D(pool_size=max_sequence_length - filter_sizes[2] + 1)(conv_2)
	maxpool_3 = MaxPool1D(pool_size=max_sequence_length - filter_sizes[3] + 1)(conv_3)
	maxpool_4 = MaxPool1D(pool_size=max_sequence_length - filter_sizes[4] + 1)(conv_4)

	z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
	z = Flatten()(z)
	z = Dropout(0.1)(z)

	outp = Dense(len(mlb.classes_), activation='sigmoid')(z)

	model = Model(inputs=inp, outputs=outp)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	return model

def textcnn_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, w2v_size, w2v_window, w2v_min_count, w2v_negative, batch_size, epochs, max_sequence_length, num_filter, drop_rate):
	EMBEDDING_DIM = int(w2v_size)

	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	train_data = dl.get_train_data(train_data_id)
	train_data = train_data_cut(train_data)

	textcnn_mlb = get_label_model(train_data['label'])
	print('数据行数:', len(train_data['label']))

	mlb_content = pickle.dumps(textcnn_mlb)
	dl.get_api_ticket()
	textcnn_mlb_id = dl.upload_file_by_data(mlb_content)
	print('textcnn_mlb_id:', textcnn_mlb_id)

	y_train = textcnn_mlb.transform(train_data['label'])

	w2v_model = get_word2vec_model(train_data['context'], int(w2v_size), int(w2v_window), int(w2v_min_count), int(w2v_negative))
	MAX_NB_WORDS = len(list(w2v_model.wv.vocab))

	tokenizer = get_word_index(MAX_NB_WORDS, train_data['context'])
	tokenizer_content = pickle.dumps(tokenizer)
	dl.get_api_ticket()
	textcnn_tokenizer_id = dl.upload_file_by_data(tokenizer_content)
	print('textcnn_tokenizer_id', textcnn_tokenizer_id)

	train_word_seq = tokenizer.texts_to_sequences(train_data['context'])
	word_index = tokenizer.word_index

	embeddings_index = get_word_vector(w2v_model)
	nb_words = min(MAX_NB_WORDS, len(word_index))
	print('nb_words:', nb_words)

	word_embedding_matrix = get_word_embedding_matrix(nb_words, EMBEDDING_DIM, word_index, MAX_NB_WORDS, embeddings_index)
	x_train = pad_sequences(train_word_seq, maxlen=int(max_sequence_length))
	print("Shape of word train data tensor:", x_train.shape)

	model = get_textcnn_model(int(max_sequence_length), nb_words, EMBEDDING_DIM, word_embedding_matrix, float(drop_rate), int(num_filter), textcnn_mlb)

	Jaccard = JaccardEvaluation(validation_data=(x_train, y_train), interval=1)

	model.fit(x_train, y_train, batch_size=int(batch_size), epochs=int(epochs), validation_data=(x_train, y_train),
			  callbacks=[Jaccard], verbose=2)

	make_keras_picklabel()
	textcnn_model = load_model('./model/best_textcnn_model.h5')

	textcnn_model_content = pickle.dumps(textcnn_model)
	dl.get_api_ticket()
	textcnn_model_id = dl.upload_file_by_data(textcnn_model_content)
	print('best_textcnn_model_id', textcnn_model_id)

	result = "mlb_id#tokenizer_id#model_id##" + str(textcnn_mlb_id) + "#" + str(textcnn_tokenizer_id) + "#" + str(textcnn_model_id)
	
	return result

def textcnn_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, textcnn_mlb_id, textcnn_tokenizer_id, textcnn_model_id, max_sequence_length, batch_size):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	data = dl.get_train_data(data_id)
	data = predict_data_cut(data)

	dl.get_api_ticket()
	textcnn_mlb = pickle.loads(dl.download_model_by_id(textcnn_mlb_id))
	print(len(textcnn_mlb.classes_))

	dl.get_api_ticket()
	textcnn_tokenizer = pickle.loads(dl.download_model_by_id(textcnn_tokenizer_id))

	dl.get_api_ticket()
	textcnn_model = pickle.loads(dl.download_model_by_id(textcnn_model_id))

	train_word_seq = textcnn_tokenizer.texts_to_sequences(data['context'])

	x_train = pad_sequences(train_word_seq, maxlen=int(max_sequence_length))
	print("Shape of word train data tensor:", x_train.shape)

	y_pred = textcnn_model.predict(x_train, batch_size=int(batch_size))
	y_pred = (y_pred > 0.5)
	# print(y_pred)

	label = [','.join(i) for i in textcnn_mlb.inverse_transform(y_pred)]
	# print(label)

	ans = {'id': data['id'], 'label': label}
	ans = pd.DataFrame(ans)
	# print(ans)

	ans['result'] = ans['id'].map(str) + "#" + ans['label'].map(str)
	result_list = ans['result'].tolist()
	
	result = ''
	for i in result_list:
		result = i + "##" + result

	return result[:-2]

if __name__ == '__main__':
	
	train_id = "87ea99c4d11411e8be95000c2961e520"
	w2v_size = 300
	w2v_window = 5
	w2v_min_count = 1
	w2v_negative = 5
	batch_size = 128
	epochs = 1
	max_sequence_length = 5000
	num_filter = 128
	drop_rate = 0.4

	result = textcnn_train(train_data_id, w2v_size, w2v_window, w2v_min_count, w2v_negative, batch_size, epochs, max_sequence_length, num_filter, drop_rate)
	print("训练返回结果：", result)