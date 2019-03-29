# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年7月10日
    文件名：	SentenceParse.py
    功能：	利用LTP模型进行语句分析，包含
    			- 分词
    			- 词性标注
    			- 句法分析
"""

import os

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')

import sys

sys.path.append(father_path)

import config_TM
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
from TemplateMatching.MyLog import Logger


class Senten_Parse:

	def __init__(self, flag=True):
		self.flag = flag  # 是否使用自定义词典，True:使用，False:不使用，默认为True
		self.segmentor = Segmentor()  # 分词模型初始化
		self.postagger = Postagger()  # 词性标注模型初始化
		self.parser = Parser()  # 句法分析模型初始化

		if self.flag:
			if os.path.exists(config_TM.userdict_path):
				# 加载分词模型，使用自定义词典
				self.segmentor.load_with_lexicon(config_TM.cws_model_path, config_TM.userdict_path)
				Logger.log_DEBUG.info('分词模型加载成功，使用自定义词典')
			else:
				Logger.log_ERROR.error('没找到自定义词典文件，请检查路径是否正确')
		else:
			# 加载分词模型，不使用自定义词典
			self.segmentor.load(config_TM.cws_model_path)
			Logger.log_DEBUG.info('分词模型加载成功，不使用自定义词典')

		# 加载词性标注模型
		self.postagger.load(config_TM.pos_model_path)
		Logger.log_DEBUG.info('词性标注模型加载成功')

		# 加载句法分析模型
		self.parser.load(config_TM.par_model_path)
		Logger.log_DEBUG.info('句法分析模型加载成功')

	def sentence_parse(self, sentence):
		"""
		语句分析
		:param sentence: 待处理语句
		:return: 返回分词结果(list)、词性标注结果(list)、句法分析结果(list-tuple)
		"""
		segmentor = self.segmentor
		postagger = self.postagger
		parser = self.parser

		# 分词结果列表
		words_list = list(segmentor.segment(sentence))

		# 词性标注
		postags = postagger.postag(words_list)
		# 词性标注结果列表
		pos_list = [pos for word, pos in zip(words_list, postags)]

		# 句法分析
		arcs = parser.parse(words_list, postags)
		# 句法分析结果列表
		arcs_list = []

		temp_arcs_list = [(arc.head, arc.relation) for arc in arcs]
		arcslist_dic = dict((i, c) for i, c in enumerate(temp_arcs_list))

		words_dic = dict((i, c) for i, c in enumerate(words_list))

		for key in arcslist_dic:
			arcslist_dic_key = arcslist_dic[key]
			if arcslist_dic_key[1] == 'HED':
				temp_list = [words_dic[key], arcslist_dic_key[1], words_dic[key]]
				arcs_list.append(temp_list)
			else:
				temp_list = [words_dic[key], arcslist_dic_key[1], words_dic[arcslist_dic_key[0] - 1]]
				arcs_list.append(temp_list)

		Logger.log_DEBUG.info('语句分析完成！')

		return words_list, pos_list, arcs_list

	def __del__(self):
		"""
		释放模型
		:return:
		"""
		self.segmentor.release() # 分词模型释放
		self.postagger.release() # 词性标注模型释放
		self.parser.release() # 句法分析模型释放
		print('-------')
		print('模型释放完成')


if __name__ == '__main__':
	SP = Senten_Parse()
	sentence = '贷记卡申请进度查询'
	words_list, pos_list, arcs_list = SP.sentence_parse(sentence)
	print(words_list)
	print(pos_list)
	print(arcs_list)

	# import xlrd
	#
	# data = xlrd.open_workbook('../data/hr_senten.xlsx')
	# table = data.sheets()[0]  # 按sheet索引号读取数据
	# nrows = table.nrows  # 所有行数
	#
	# for rownum in range(10):
	# 	senten = table.row_values(rownum)[0]
	#
	# 	words_list, pos_list, arcs_list = SP.sentence_parse(senten)
	# 	print(words_list)
	# 	print(pos_list)
	# 	print(arcs_list)
