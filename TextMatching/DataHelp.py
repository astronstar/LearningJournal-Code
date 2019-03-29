# coding:utf-8

import numpy as np
import pandas as pd
import jieba
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import word2vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer


def save_pkl(file, object):
	with open(file, 'wb') as file:
		pickle.dump(object, file)

def load_pkl(file):
	with open(file, 'rb') as file:
		result = pickle.load(file)
		return result

def load_stopwords(file):
	with open(file, 'r', encoding='utf-8') as file:
		stop_words = [line.strip().replace('\n', '') for line in file]
		return stop_words

stopwords = load_stopwords('./data/stop_words.txt')
jieba.load_userdict('./data/dict_all.txt')

def seg_sentence(sentence):
	sentence_seged = jieba.cut(sentence.strip())

	result = ""
	for word in sentence_seged:
		if word not in stopwords:
			if word != ' ' and word != '':
				result += word + ' '

	result = result.strip()

	return result

def sentence_filter(sentence):
	sentence_list = list(sentence)

	result = ''
	for r in sentence_list:
		if r not in stopwords:
			result += r + ' '
	result = result.strip()

	return result

def data_part(data_file):
	df = pd.read_csv(data_file, sep='\t', names=['id', 's1', 's2', 'label'])

	df['s1_seg'] = df['s1'].apply(seg_sentence)
	df['s2_seg'] = df['s2'].apply(seg_sentence)

	df['s1_char'] = df['s1'].apply(sentence_filter)
	df['s2_char'] = df['s2'].apply(sentence_filter)

	X_train_1, x_test, y_train_1, y_test = train_test_split(
		df[['id', 's1', 's2', 's1_seg', 's2_seg', 's1_char', 's2_char']], df[['label']], test_size=0.1, random_state=42)

	x_train, x_val, y_train, y_val = train_test_split(X_train_1, y_train_1, test_size=0.2, random_state=42)

	x_train.to_csv('./data/preprocessed/x_train.csv', index=None)
	x_test.to_csv('./data/preprocessed/x_test.csv', index=None)
	x_val.to_csv('./data/preprocessed/x_val.csv', index=None)

	y_train.to_csv('./data/preprocessed/y_train.csv', index=None)
	y_test.to_csv('./data/preprocessed/y_test.csv', index=None)
	y_val.to_csv('./data/preprocessed/y_val.csv', index=None)

	train_label = y_train['label'].tolist()
	train_label = np.array(train_label).reshape(-1, 1)

	test_label = y_test['label'].tolist()
	test_label = np.array(test_label).reshape(-1, 1)

	val_label = y_val['label'].tolist()
	val_label = np.array(val_label).reshape(-1, 1)

	save_pkl('./data/preprocessed/y_train.pkl', train_label)
	save_pkl('./data/preprocessed/y_test.pkl', test_label)
	save_pkl('./data/preprocessed/y_val.pkl', val_label)
	print('finished save label_unit')

	print('==== finished data_part and preprocessing ====')


def get_tfidf_model():
	train_df = pd.read_csv('./data/preprocessed/x_train.csv')
	train_df = train_df.fillna(' ')

	texts_word = []
	s1_texts_word = train_df['s1_seg'].tolist()
	s2_texts_word = train_df['s2_seg'].tolist()
	texts_word.extend(s1_texts_word)
	texts_word.extend(s2_texts_word)

	texts_char = []
	s1_texts_char = train_df['s1_char'].tolist()
	s2_texts_char = train_df['s2_char'].tolist()
	texts_char.extend(s1_texts_char)
	texts_char.extend(s2_texts_char)
	texts_char = [i for i in texts_char if len(i) != 0]

	tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
	tfidf_word.fit(texts_word)
	save_pkl('./data/model/tfidf_word_model.pkl', tfidf_word)
	print('finished tfidf_word model')

	tfidf_char = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), max_features=6000)
	tfidf_char.fit(texts_char)
	save_pkl('./data/model/tfidf_char_model.pkl', tfidf_char)
	print('finished tfidf_char model')

	print('==== finished get_tfidf_model ====')


def get_train_w2v():
	train_df = pd.read_csv('./data/preprocessed/x_train.csv')
	train_df = train_df.fillna(' ')

	train_df['s1_seg'] = train_df['s1_seg'].apply(lambda x: x.split(' '))
	train_df['s2_seg'] = train_df['s2_seg'].apply(lambda x: x.split(' '))

	s1_texts = train_df['s1_seg'].tolist()
	s2_texts = train_df['s2_seg'].tolist()

	texts = []
	texts.extend(s1_texts)
	texts.extend(s2_texts)

	w2v_model_skip = word2vec.Word2Vec(sentences=texts, size=300, sg=1, min_count=3, window=2)
	w2v_model_skip.wv.save_word2vec_format('./data/model/train_word_w2v_skip.bin', binary=True)
	print('finished get_train_w2v skip')

	w2v_model_cbow = word2vec.Word2Vec(sentences=texts, size=300, sg=0, min_count=3, window=2)
	w2v_model_cbow.wv.save_word2vec_format('./data/model/train_word_w2v_cbow.bin', binary=True)
	print('finished get_train_w2v cbow')

	print('==== finished get_train_w2v ====')


def get_train_char_w2v():
	train_df = pd.read_csv('./data/preprocessed/x_train.csv')
	train_df = train_df.fillna(' ')

	train_df['s1_char'] = train_df['s1_char'].apply(lambda x: x.split(' '))
	train_df['s2_char'] = train_df['s2_char'].apply(lambda x: x.split(' '))

	s1_char_texts = train_df['s1_char'].tolist()
	s2_char_texts = train_df['s2_char'].tolist()

	char_texts = []
	char_texts.extend(s1_char_texts)
	char_texts.extend(s2_char_texts)

	w2v_model_skip = word2vec.Word2Vec(sentences=char_texts, size=300, sg=1, window=3)
	w2v_model_skip.wv.save_word2vec_format('./data/model/train_char_w2v_skip.bin', binary=True)
	print('finished get_train_char_w2v skip')

	w2v_model_cbow = word2vec.Word2Vec(sentences=char_texts, size=300, sg=0, window=3)
	w2v_model_cbow.wv.save_word2vec_format('./data/model/train_char_w2v_cbow.bin', binary=True)
	print('finished get_train_char_w2v cbow')

	print('==== finished get_train_char_w2v ====')


def save_embedding_w2v(MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SENTENCT_LENGTH):
	print('load skip w2v_model...')
	skip_w2v_path = './data/model/train_word_w2v_skip.bin'
	skip_w2v_model = KeyedVectors.load_word2vec_format(skip_w2v_path, binary=True, encoding='utf8')
	print('finish skip w2v_model...')

	print('load cbow w2v_model...')
	cbow_w2v_path = './data/model/train_word_w2v_cbow.bin'
	cbow_w2v_model = KeyedVectors.load_word2vec_format(cbow_w2v_path, binary=True, encoding='utf8')
	print('finish cbow w2v_model...')

	train_df = pd.read_csv('./data/preprocessed/x_train.csv')
	train_df = train_df.fillna(' ')
	texts = []
	s1_texts = train_df['s1_seg'].tolist()
	s2_texts = train_df['s2_seg'].tolist()
	texts.extend(s1_texts)
	texts.extend(s2_texts)

	tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
	tokenizer.fit_on_texts(texts)
	save_pkl('./data/model/train_tokenizer.pkl', tokenizer)

	word_index = tokenizer.word_index
	num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

	embedding_matrix_skip = np.zeros((num_words, EMBEDDING_DIM))
	for word, index in word_index.items():
		if word in skip_w2v_model.wv.vocab:
			embedding_matrix_skip[index] = skip_w2v_model.word_vec(word)
	save_pkl('./data/model/train_word_w2v_embedding_matrix_skip.pkl', embedding_matrix_skip)
	print('finished save embedding_matrix_skip')

	embedding_matrix_cbow = np.zeros((num_words, EMBEDDING_DIM))
	for word, index in word_index.items():
		if word in cbow_w2v_model.wv.vocab:
			embedding_matrix_cbow[index] = cbow_w2v_model.word_vec(word)
	save_pkl('./data/model/train_word_w2v_embedding_matrix_cbow.pkl', embedding_matrix_cbow)
	print('finished save embedding_matrix_cbow')

	s1_train_ids = tokenizer.texts_to_sequences(s1_texts)
	s2_train_ids = tokenizer.texts_to_sequences(s2_texts)

	s1_train_ids_pad = sequence.pad_sequences(s1_train_ids, maxlen=MAX_SENTENCT_LENGTH)
	s2_train_ids_pad = sequence.pad_sequences(s2_train_ids, maxlen=MAX_SENTENCT_LENGTH)

	save_pkl('./data/preprocessed/s1_train_ids_pad.pkl', s1_train_ids_pad)
	save_pkl('./data/preprocessed/s2_train_ids_pad.pkl', s2_train_ids_pad)
	print('finished save train_ids_pad')

	# 存储验证集相关数据
	val_df = pd.read_csv('./data/preprocessed/x_val.csv')
	val_df = val_df.fillna(' ')
	s1_val_texts = val_df['s1_seg'].tolist()
	s2_val_texts = val_df['s2_seg'].tolist()

	s1_val_ids = tokenizer.texts_to_sequences(s1_val_texts)
	s2_val_ids = tokenizer.texts_to_sequences(s2_val_texts)

	s1_val_ids_pad = sequence.pad_sequences(s1_val_ids, maxlen=MAX_SENTENCT_LENGTH)
	s2_val_ids_pad = sequence.pad_sequences(s2_val_ids, maxlen=MAX_SENTENCT_LENGTH)

	save_pkl('./data/preprocessed/s1_val_ids_pad.pkl', s1_val_ids_pad)
	save_pkl('./data/preprocessed/s2_val_ids_pad.pkl', s2_val_ids_pad)
	print('finished save val_ids_pad')

	# 存储测试集相关数据
	test_df = pd.read_csv('./data/preprocessed/x_test.csv')
	test_df = test_df.fillna(' ')
	s1_test_texts = test_df['s1_seg'].tolist()
	s2_test_texts = test_df['s2_seg'].tolist()

	s1_test_ids = tokenizer.texts_to_sequences(s1_test_texts)
	s2_test_ids = tokenizer.texts_to_sequences(s2_test_texts)

	s1_test_ids_pad = sequence.pad_sequences(s1_test_ids, maxlen=MAX_SENTENCT_LENGTH)
	s2_test_ids_pad = sequence.pad_sequences(s2_test_ids, maxlen=MAX_SENTENCT_LENGTH)

	save_pkl('./data/preprocessed/s1_test_ids_pad.pkl', s1_test_ids_pad)
	save_pkl('./data/preprocessed/s2_test_ids_pad.pkl', s2_test_ids_pad)
	print('finished save test_ids_pad')

	print('==== finished save_embedding_w2v ====')


def save_char_embedding_w2v(MAX_NUM_WORDS, EMBEDDING_DIM, MAX_WORD_LENGTH):
	print('load skip char_w2v_model...')
	skip_w2v_path = './data/model/train_char_w2v_skip.bin'
	skip_w2v_model = KeyedVectors.load_word2vec_format(skip_w2v_path, binary=True, encoding='utf8')
	print('finish skip char_w2v_model...')

	print('load cbow char_w2v_model...')
	cbow_w2v_path = './data/model/train_char_w2v_cbow.bin'
	cbow_w2v_model = KeyedVectors.load_word2vec_format(cbow_w2v_path, binary=True, encoding='utf8')
	print('finish cbow char_w2v_model...')

	train_df = pd.read_csv('./data/preprocessed/x_train.csv')
	train_df = train_df.fillna(' ')
	char_texts = []
	s1_char_texts = train_df['s1_char'].tolist()
	s2_char_texts = train_df['s2_char'].tolist()
	char_texts.extend(s1_char_texts)
	char_texts.extend(s2_char_texts)

	tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
	tokenizer.fit_on_texts(char_texts)
	save_pkl('./data/model/train_char_tokenizer.pkl', tokenizer)

	word_index = tokenizer.word_index
	num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

	embedding_matrix_skip = np.zeros((num_words, EMBEDDING_DIM))
	for word, index in word_index.items():
		if word in skip_w2v_model.wv.vocab:
			embedding_matrix_skip[index] = skip_w2v_model.word_vec(word)
	save_pkl('./data/model/train_char_w2v_embedding_matrix_skip.pkl', embedding_matrix_skip)
	print('finished save char embedding_w2v skip')

	embedding_matrix_cbow = np.zeros((num_words, EMBEDDING_DIM))
	for word, index in word_index.items():
		if word in cbow_w2v_model.wv.vocab:
			embedding_matrix_cbow[index] = cbow_w2v_model.word_vec(word)
	save_pkl('./data/model/train_char_w2v_embedding_matrix_cbow.pkl', embedding_matrix_cbow)
	print('finished save char embedding_w2v cbow')

	s1_train_char_ids = tokenizer.texts_to_sequences(s1_char_texts)
	s2_train_char_ids = tokenizer.texts_to_sequences(s2_char_texts)

	s1_train_char_ids_pad = sequence.pad_sequences(s1_train_char_ids, maxlen=MAX_WORD_LENGTH)
	s2_train_char_ids_pad = sequence.pad_sequences(s2_train_char_ids, maxlen=MAX_WORD_LENGTH)

	save_pkl('./data/preprocessed/s1_train_char_ids_pad.pkl', s1_train_char_ids_pad)
	save_pkl('./data/preprocessed/s2_train_char_ids_pad.pkl', s2_train_char_ids_pad)
	print('finished save train_char_ids_pad')

	# 存储验证集相关数据
	val_df = pd.read_csv('./data/preprocessed/x_val.csv')
	val_df = val_df.fillna(' ')
	s1_val_char_texts = val_df['s1_char'].tolist()
	s2_val_char_texts = val_df['s2_char'].tolist()

	s1_val_char_ids = tokenizer.texts_to_sequences(s1_val_char_texts)
	s2_val_char_ids = tokenizer.texts_to_sequences(s2_val_char_texts)

	s1_val_char_ids_pad = sequence.pad_sequences(s1_val_char_ids, maxlen=MAX_WORD_LENGTH)
	s2_val_char_ids_pad = sequence.pad_sequences(s2_val_char_ids, maxlen=MAX_WORD_LENGTH)

	save_pkl('./data/preprocessed/s1_val_char_ids_pad.pkl', s1_val_char_ids_pad)
	save_pkl('./data/preprocessed/s2_val_char_ids_pad.pkl', s2_val_char_ids_pad)
	print('finished save val_char_ids_pad')

	# 存储测试集相关数据
	test_df = pd.read_csv('./data/preprocessed/x_test.csv')
	test_df = test_df.fillna(' ')
	s1_test_char_texts = test_df['s1_char'].tolist()
	s2_test_char_texts = test_df['s2_char'].tolist()

	s1_test_char_ids = tokenizer.texts_to_sequences(s1_test_char_texts)
	s2_test_char_ids = tokenizer.texts_to_sequences(s2_test_char_texts)

	s1_test_char_ids_pad = sequence.pad_sequences(s1_test_char_ids, maxlen=MAX_WORD_LENGTH)
	s2_test_char_ids_pad = sequence.pad_sequences(s2_test_char_ids, maxlen=MAX_WORD_LENGTH)

	save_pkl('./data/preprocessed/s1_test_char_ids_pad.pkl', s1_test_char_ids_pad)
	save_pkl('./data/preprocessed/s2_test_char_ids_pad.pkl', s2_test_char_ids_pad)
	print('finished save test_char_ids_pad')

	print('==== finished save_char_embedding_w2v ====')


if __name__ == '__main__':
	all_data = './data/atec_nlp_sim_train_all.csv'

	data_part(all_data)

	get_tfidf_model()

	get_train_w2v()
	get_train_char_w2v()

	MAX_NUM_WORDS = 10000
	EMBEDDING_DIM = 300
	MAX_SENTENCT_LENGTH = 20
	MAX_WORD_LENGTH = 24

	save_embedding_w2v(MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SENTENCT_LENGTH)
	save_char_embedding_w2v(MAX_NUM_WORDS, EMBEDDING_DIM, MAX_WORD_LENGTH)
