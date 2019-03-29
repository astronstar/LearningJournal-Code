# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年9月17日
    文件名：	IndivWord_Replace.py
    功能：	相似语句个体词替换
"""

import os

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')

import sys

sys.path.append(father_path)

import jieba
import Levenshtein

import config_TM
from TemplateMatching.load_data import LoadData
from TemplateMatching.MyLog import Logger


class WordReplace:
	def __init__(self):
		self.LD = LoadData([], {}, 'E:/stopword.txt', 'E:/test.txt')

	def senten_seg(self, sentence, flag=True):
		if flag:
			stopword = self.LD.stopword

			sentence_seg = list(jieba.cut(sentence))
			result = []
			for word in sentence_seg:
				if word not in stopword:
					result.append(word)

			return result
		else:
			result = list(jieba.cut(sentence))

			return result

	def list_compare(self, l1, l2):
		l1_word = self.senten_seg(l1)
		l2_word = self.senten_seg(l2)

		if len(l1_word) == len(l2_word):
			# l1_str = ''.join(l1_word)
			# l2_str = ''.join(l2_word)
			#
			# result = Levenshtein.jaro(l1_str, l2_str)
			#
			# if result <= 0.6:
			# 	flag = True
			# else:
			# 	flag = False

			temp = list(set(l1_word) & set(l2_word))
			if len(temp) == len(l1_word)-1:
				w1 = list(set(l1_word) - set(l2_word))[0]
				w2 = list(set(l2_word) - set(l1_word))[0]
				flag = True
			else:
				w1, w2 = '', ''
				flag = False
		else:
			w1, w2 = '', ''
			flag = False

		return w1, w2, flag

	def word_replace(self, sentence):
		senten_dic = self.LD.senten
		result = {}
		for s in senten_dic:
			w1, w2, flag = self.list_compare(sentence, s)
			if flag:
				s_value = senten_dic[s]
				for r in s_value:
					r_list = self.senten_seg(r,flag=False)
					r_dic = dict((i,c) for i,c in enumerate(r_list))
					for i in r_dic:
						if r_dic[i] == w2:
							r_dic[i] = w1
					temp = ''.join(r_dic.values())
					result.setdefault(sentence, []).append(temp)
			else:
				continue

		return result

if __name__ == '__main__':

	WP = WordReplace()

	l1 = '查询信用卡的积分'

	r = WP.word_replace(l1)
	print(r)


