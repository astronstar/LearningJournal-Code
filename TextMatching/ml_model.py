# coding:utf-8

import jieba
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_similarity_score

class ML_Model:

	def __init__(self, tfidf_word_path, tfidf_char_path, w2v_word_path, w2v_char_path):
		# self.w2v_model = KeyedVectors.load_word2vec_format('./data/preprocessed/train_w2v_skip.bin', binary=True, encoding='utf8')
		# self.tfidf_model = load_pkl('./data/preprocessed/tfidf_model.pkl')
		self.w2v_word_model = KeyedVectors.load_word2vec_format(w2v_word_path, binary=True, encoding='utf8')
		self.w2v_char_model = KeyedVectors.load_word2vec_format(w2v_char_path, binary=True, encoding='utf8')
		# self.tfidf_word_model = load_pkl(tfidf_word_path)
		# self.tfidf_char_model = load_pkl(tfidf_char_path)
		jieba.load_userdict('./data/dict_all.txt')
		self.stopwords = self.load_stopwords('./data/stop_words.txt')

	def load_stopwords(self, file):
		with open(file, 'r', encoding='utf-8') as file:
			stop_words = [line.strip().replace('\n', '') for line in file]
			return stop_words

	def seg_sentence(self, sentence):
		sentence_seged = jieba.cut(sentence.strip())

		result = ""
		for word in sentence_seged:
			if word not in self.stopwords:
				if word != ' ' and word != '':
					result += word + ' '

		result = result.strip()

		return result

	def sentence_filter(self, sentence):
		sentence_list = list(sentence)

		result = ''
		for r in sentence_list:
			if r not in self.stopwords:
				result += r + ' '
		result = result.strip()

		return result

	def get_tfidf_word_vector(self, sentence):
		tfidf_word_model = self.tfidf_word_model

		texts = []
		for s in sentence:
			s_seg = self.seg_sentence(s)
			texts.append(s_seg)

		senten_vector = tfidf_word_model.fit_transform(texts).todense()

		s1_vector = np.array(senten_vector[0])
		s2_vector = np.array(senten_vector[1])
		result = (s1_vector, s2_vector)

		return result

	def get_tfidf_char_vector(self, sentence):
		tfidf_char_model = self.tfidf_char_model

		texts = []
		for s in sentence:
			s_seg = self.sentence_filter(s)
			texts.append(s_seg)

		senten_vector = tfidf_char_model.fit_transform(texts).todense()

		s1_vector = np.array(senten_vector[0])
		s2_vector = np.array(senten_vector[1])
		result = (s1_vector, s2_vector)

		return result

	def get_w2v_word_vector(self, sentence):
		w2v_word_model = self.w2v_word_model

		senten_seg = self.seg_sentence(sentence)
		senten_list = senten_seg.split(' ')

		num = len(senten_list)
		senten_vector = np.zeros((num, 300))

		for i in range(num):
			if senten_list[i] in w2v_word_model.wv.vocab:
				senten_vector[i] = w2v_word_model.word_vec(senten_list[i])
		result = np.sum(senten_vector, axis=0) / num
		result = result.reshape(1, -1)

		return result

	def get_w2v_char_vector(self, sentence):
		w2v_char_model = self.w2v_char_model

		senten_char = self.sentence_filter(sentence)
		senten_list = senten_char.split(' ')

		num = len(senten_list)
		senten_vector = np.zeros((num, 300))

		for i in range(num):
			if senten_list[i] in w2v_char_model.wv.vocab:
				senten_vector[i] = w2v_char_model.word_vec(senten_list[i])

		result = np.sum(senten_vector, axis=0) / num
		result = result.reshape(1, -1)

		return result

	def cosine_distance(self, s1, s2):
		try:
			result = np.dot(s1, s2.T) / (np.sqrt(np.sum(s1 ** 2)) * np.sqrt(np.sum(s2 ** 2)))
			result = result[0][0]
			return result
		except Exception as e:
			print(str(e))
			return 0.0

	def calc_cosine_similar(self, sentence, mode='w2v_word'):
		if mode == 'tfidf_word':
			tfidf_vector = self.get_tfidf_word_vector(sentence)

			result = self.cosine_distance(tfidf_vector[0], tfidf_vector[1])

			return result
		elif mode == 'tfidf_char':
			tfidf_vector = self.get_tfidf_char_vector(sentence)

			result = self.cosine_distance(tfidf_vector[0], tfidf_vector[1])

			return result
		elif mode == 'w2v_word':
			s1_w2v_vector = self.get_w2v_word_vector(sentence[0])
			s2_w2v_vector = self.get_w2v_word_vector(sentence[1])

			result = self.cosine_distance(s1_w2v_vector, s2_w2v_vector)

			return result
		elif mode == 'w2v_char':
			s1_w2v_vector = self.get_w2v_char_vector(sentence[0])
			s2_w2v_vector = self.get_w2v_char_vector(sentence[1])

			result = self.cosine_distance(s1_w2v_vector, s2_w2v_vector)

			return result

	def get_jaccard_simlar(self, sentence):
		s1_seg = self.seg_sentence(sentence[0])
		s1_list = s1_seg.split(' ')

		s2_seg = self.seg_sentence(sentence[1])
		s2_list = s2_seg.split(' ')

		unions = set(s1_list).union(set(s2_list))
		intersections = set(s1_list).intersection(set(s2_list))

		result = len(intersections) / len(unions)

		return result

	@staticmethod
	def r_f1_thresh(y_pred, y_true, step=1000):
		e = np.zeros((len(y_true), 2))

		e[:, 0] = y_pred.reshape(1, -1)
		e[:, 1] = y_true.reshape(1, -1)

		f = pd.DataFrame(e)
		thrs = np.linspace(0, 1, step+1)
		x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:,1]) for thr in thrs])
		f1_, thresh = max(x), thrs[x.argmax()]

		return f.corr()[0][1], f1_, thresh

	@staticmethod
	def get_metrics(y_test, predictions):
		print('accuracy_score', accuracy_score(y_test, predictions))
		print('precision_score', precision_score(y_test, predictions))
		print('recall_score', recall_score(y_test, predictions))
		print('f1_score', f1_score(y_test, predictions))


if __name__ == '__main__':

	tfidf_word_path = './data/model/tfidf_word_model.pkl'
	tfidf_char_path = './data/model/tfidf_char_model.pkl'
	w2v_word_path = './data/model/train_word_w2v_skip.bin'
	w2v_char_path = './data/model/train_char_w2v_skip.bin'

	ml_model = ML_Model(tfidf_word_path, tfidf_char_path, w2v_word_path, w2v_char_path)

	x_test = pd.read_csv('./data/preprocessed/x_test.csv')
	y_test = pd.read_csv('./data/preprocessed/y_test.csv')

	s1 = x_test['s1'].tolist()
	s2 = x_test['s2'].tolist()
	y_label = y_test['label'].tolist()
	y_true = np.array(y_label)

	test_texts = zip(s1, s2)

	result1 = []
	result2 = []
	result3 = []
	for i in test_texts:
		similar1 = ml_model.calc_cosine_similar(i, mode='w2v_word')
		similar2 = ml_model.calc_cosine_similar(i, mode='w2v_char')
		similar3 = ml_model.get_jaccard_simlar(i)

		result1.append(similar1)
		result2.append(similar2)
		result3.append(similar3)

	result1_out = np.array(result1)
	result2_out = np.array(result2)
	result3_out = np.array(result3)

	r1_1, r2_1, r3_1 = ml_model.r_f1_thresh(result1_out, y_true)
	print('==== w2v_word计算结果 ====')
	print('R相关系数:', r1_1, '最优评分:', r2_1, 'f1阈值:', r3_1)
	ml_model.get_metrics(y_label, (result1_out >= r3_1))

	r1_2, r2_2, r3_2 = ml_model.r_f1_thresh(result2_out, y_true)
	print('==== w2v_char计算结果 ====')
	print('R相关系数:', r1_2, '最优评分:', r2_2, 'f1阈值:', r3_2)
	ml_model.get_metrics(y_label, (result2_out >= r3_2))

	r1_3, r2_3, r3_3 = ml_model.r_f1_thresh(result3_out, y_true)
	print('==== jaccard_similary计算结果 ====')
	print('R相关系数:', r1_3, '最优评分:', r2_3, 'f1阈值:', r3_3)
	ml_model.get_metrics(y_label, (result3_out >= r3_3))
