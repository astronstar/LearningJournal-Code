# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年7月10日
    文件名：	SentenceRepeat.py
    功能：	复述结果生成，包含
    			- 复述模板获取
    			- 复述结果生成
"""

import os
pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')

import sys
sys.path.append(father_path)

import config_TM
from TemplateMatching.SentenceParse import Senten_Parse
from TemplateMatching.InitializeDict import Initial_Dict_Load
from TemplateMatching.MyLog import Logger
import copy
from itertools import permutations


class Senten_Repeat:
	def __init__(self):
		# 实例化相关词典
		self.Init_Dic = Initial_Dict_Load({}, {}, {}, {},
		                                  config_TM.indivword_path, config_TM.featureword_path, config_TM.template_path)

		# 合并个体词-标签词典和功能词-标签词典
		self.ReplacePos_Dic = dict(self.Init_Dic.IndivWord_Dic, **self.Init_Dic.WordTag_Dic)

		self.SP = Senten_Parse()

	def list_to_dic(self, the_list):
		"""
		列表转换为字典(key:索引号, value:列表中对应元素)
		:param the_list: 待处理列表
		:return: 列表字典化结果
		"""
		listDic = dict((i, c) for i, c in enumerate(the_list))

		return listDic

	def cal_freq(self, word, arcs_dic):
		"""
		计算句法分析结果arcs_dic中与word有关系的词的个数
		:param word: 待计算词
		:param arcs_dic: 句法分析字典式结果
		:return: 个数(int格式)
		"""
		num = 0
		for i in arcs_dic:
			if word in arcs_dic[i]:
				num += 1

		return num

	def cal_index_diff_value(self, the_list):
		"""
		返回列表中满足条件(相邻元素的差值是否为1)的元素的索引号
		:param the_list: 输入列表
		:return: 满足条件的元素索引号(list嵌套list格式)
		"""
		result, temp_list = [], []
		for i in range(1, len(the_list)):
			if the_list[i] - the_list[i - 1] == 1:
				temp_list.append(i - 1)
				temp_list.append(i)
			else:
				if len(temp_list) > 0:
					# result.extend(sorted(list(set(temp_list))))
					result.append(sorted(list(set(temp_list))))
				temp_list = []
		if len(temp_list) > 0:
			# result.extend(sorted(list(set(temp_list))))
			result.append(sorted(list(set(temp_list))))

		return result

	def get_max_element_index(self, the_list):
		"""
		获取列表中值最大的索引号
		:param the_list: 待处理数值列表
		:return: 索引号结果(list格式)
		"""
		max_value = max(the_list)
		the_list_dic = self.list_to_dic(the_list)

		result = [key for key in the_list_dic if the_list_dic[key] == max_value]

		return result

	def get_index(self, word, word_dic):
		"""
		获取某个词在字典中的键，即该词的索引号
		:param word: 待获取索引号词
		:param word_dic: 包含该词的字典式结果
		:return: 键，即索引号
		"""
		for key in word_dic:
			if word_dic[key] == word:
				word_index = key
			else:
				continue

		return word_index

	def hed_verb_norm(self, word_dic, pos_dic, arcs_dic):
		"""
		核心词确定
		:param word_dic: 分词字典式结果
		:param pos_dic: 词性标注字典式结果
		:param arcs_dic: 句法分析字典式结果
		:return: word_dic, pos_dic, arcs_dic
		"""
		for key in arcs_dic:
			arcs_dic_key = arcs_dic[key]

			if arcs_dic_key[1] == 'HED':
				if pos_dic[key] == 'indiv':
					pos_dic[key] = 'hed'
				else:
					if pos_dic[key] == 'n':
						noun_key_left = list(range(key))
						noun_key_left.reverse()

						for i in noun_key_left:
							if (word_dic[key] in arcs_dic[i]) and (pos_dic[i] == 'v'):
								pos_dic[i] = 'hed'

						pos_all = [pos_dic[i] for i in pos_dic]
						if 'hed' not in pos_all:
							for i in noun_key_left:
								if pos_dic[i] == 'v':
									pos_dic[i] = 'hed'

						arcs_dic[key] = []

					elif pos_dic[key] == 'v':
						if (key == len(arcs_dic) - 1) or (key == 0):
							pos_dic[key] = 'hed'
						else:
							verb_key_left = list(range(key))
							verb_key_left.reverse()

							# verb_key_right = list(range(key+1, len(word_dic)))
							for i in verb_key_left:
								if (word_dic[key] in arcs_dic[i]) and (pos_dic[i] == 'v'):
									pos_dic[i] = 'hed'

							# pos_all = [pos_dic[i] for i in pos_dic]
							#
							# if 'hed' not in pos_all:
							# 	for i in verb_key_right:
							# 		if (word_dic[key] in arcs_dic[i]) and (pos_dic[i] == 'v'):
							# 			pos_dic[i] ='hed'

							pos_all = [pos_dic[i] for i in pos_dic]
							if 'hed' not in pos_all:
								for i in verb_key_left:
									if pos_dic[i] == 'v':
										pos_dic[i] = 'hed'
							pos_all = [pos_dic[i] for i in pos_dic]
							if 'hed' not in pos_all:
								pos_dic[key] = 'hed'

						arcs_dic[key] = []
		return word_dic, pos_dic, arcs_dic

	def merge_indiv_norm(self, word_dic, pos_dic, arcs_dic):
		"""
		个体词合并
		:param word_dic: 分词字典式结果
		:param pos_dic: 词性标注字典式结果
		:param arcs_dic: 句法分析字典式结果
		:return: word_dic, pos_dic, arcs_dic
		"""
		# 统计所有个体词的索引号
		senten_indiv_index = [i for i in pos_dic if pos_dic[i] == 'indiv']

		if len(senten_indiv_index) > 1:
			# 统计满足个体词合并规则的索引号
			merge_indiv_index = self.cal_index_diff_value(senten_indiv_index)

			for num in range(len(merge_indiv_index)):
				index = merge_indiv_index[num]
				if len(index) != 0:
					index_dic = self.list_to_dic(index)

					# 根据索引号获取待合并的词
					merge_word = [word_dic[i] for i in index]

					# 计算每个词在句法分析中出现的次数
					word_freq = [self.cal_freq(w, arcs_dic) for w in merge_word]

					# 统计合并后应保留词的索引号
					max_index = self.get_max_element_index(word_freq)
					if len(max_index) == 1:
						num_value = index_dic[max_index[0]]
					else:
						num_value = index_dic[max_index[-1]]

					temp_word = word_dic[num_value]
					new_word = ''

					for i in index:
						new_word += word_dic[i]

					# 修改句法分析
					for key in arcs_dic:
						if temp_word in arcs_dic[key]:
							temp_dic = self.list_to_dic(arcs_dic[key])
							temp_word_index = self.get_index(temp_word, temp_dic)
							arcs_dic[key][temp_word_index] = new_word

					# 修改词性
					pos_dic[num_value] = 'indiv'

					# 修改词
					word_dic[num_value] = new_word

					index.remove(num_value)

					for i in index:
						word_dic[i] = ''
						pos_dic[i] = ''
						arcs_dic[i] = []

		return word_dic, pos_dic, arcs_dic

	def senten_simplify_1(self, the_list, word_dic, pos_dic, arcs_dic):
		"""
		句式精简第一步
		:param the_list: 索引号，不包含个体词和核心动词
		:param word_dic: 分词字典式结果
		:param pos_dic: 词性标注字典式结果
		:param arcs_dic: 句法分析字典式结果
		:return: word_dic, pos_dic, arcs_dic
		"""
		for key in pos_dic:
			if pos_dic[key] == 'hed':
				hed_index = key
			if pos_dic[key] == 'indiv':
				indiv_index = key

		if len(the_list) > 1:
			for i in range(len(the_list)):
				num_1 = the_list[i]
				if len(arcs_dic[num_1]) > 0:
					gx = arcs_dic[num_1][1]

					if (gx == 'ATT') or (gx == 'ADV') or (gx == 'SBV') or (gx == 'VOB'):
						w1 = arcs_dic[num_1][0]
						w1_index = self.get_index(w1, word_dic)
						w1_pos = pos_dic[w1_index]

						w2 = arcs_dic[num_1][2]
						w2_index = self.get_index(w2, word_dic)
						w2_pos = pos_dic[w2_index]

						if (w2_pos != 'hed') and (w2_pos != 'indiv'):
							if w1_index < w2_index:
								if (hed_index not in list(range(w1_index, w2_index))) and \
									(indiv_index not in list(range(w1_index, w2_index))):
									new_word = w1 + w2

									word_dic[w2_index] = new_word

									# 修改词性标签
									if (w1_pos == 'num') or (w2_pos == 'num'):
										pos_dic[w2_index] = 'num'
									elif (w1_pos == 'n') or (w2_pos == 'n'):
										pos_dic[w2_index] = 'n'
									elif (w1_pos == 'v') and (w2_pos == 'v'):
										pos_dic[w2_index] = 'v'
									else:
										pos_dic[w2_index] = 'v'

									# 修改句法分析
									for key in arcs_dic:
										arcs_dic_key = arcs_dic[key]

										if w1 in arcs_dic_key:
											arcs_dic_key[arcs_dic_key.index(w1)] = new_word
										if w2 in arcs_dic_key:
											arcs_dic_key[arcs_dic_key.index(w2)] = new_word

									word_dic[w1_index] = ''
									pos_dic[w1_index] = ''
									arcs_dic[w1_index] = []
								else:
									continue
							else:
								if (hed_index not in list(range(w2_index, w1_index))) and \
									(indiv_index not in list(range(w2_index, w1_index))):
									new_word = w2 + w1

									word_dic[w2_index] = new_word

									# 修改词性标签
									if (w1_pos == 'num') or (w2_pos == 'num'):
										pos_dic[w2_index] = 'num'
									elif (w1_pos == 'n') or (w2_pos == 'n'):
										pos_dic[w2_index] = 'n'
									elif (w1_pos == 'v') and (w2_pos == 'v'):
										pos_dic[w2_index] = 'v'
									else:
										pos_dic[w2_index] = 'v'

									# 修改句法分析
									for key in arcs_dic:
										arcs_dic_key = arcs_dic[key]

										if w1 in arcs_dic_key:
											arcs_dic_key[arcs_dic_key.index(w1)] = new_word
										if w2 in arcs_dic_key:
											arcs_dic_key[arcs_dic_key.index(w2)] = new_word

									word_dic[w1_index] = ''
									pos_dic[w1_index] = ''
									arcs_dic[w1_index] = []
								else:
									continue
			return word_dic, pos_dic, arcs_dic
		else:
			return word_dic, pos_dic, arcs_dic

	def senten_simplify_2(self, word_dic, pos_dic, arcs_dic):
		"""
		句式精简第二步
		:param word_dic: 分词字典式结果
		:param pos_dic: 词性标注字典式结果
		:param arcs_dic: 句法分析字典式结果
		:return: word_dic, pos_dic, arcs_dic
		"""
		for key in pos_dic:
			if pos_dic[key] == 'hed':
				hed_word = word_dic[key]
				hed_word_index = key

		contain_hed_verb_index = []

		temp_left_list = []
		for key in range(hed_word_index):
			if len(arcs_dic[key]) > 0:
				if (pos_dic[key] != 'indiv') and (arcs_dic[key][2] == hed_word):
					temp_left_list.append(key)
		if len(temp_left_list) > 0:
			contain_hed_verb_index.append(temp_left_list)

		temp_right_list = []
		for key in range(hed_word_index + 1, len(arcs_dic)):
			if len(arcs_dic[key]) > 0:
				if (pos_dic[key] != 'indiv') and (arcs_dic[key][2] == hed_word):
				# if (arcs_dic[key][2] == hed_word):
					temp_right_list.append(key)
		if len(temp_right_list) > 0:
			contain_hed_verb_index.append(temp_right_list)

		for i in range(len(contain_hed_verb_index)):
			merge_word = [word_dic[j] for j in contain_hed_verb_index[i]]

			merge_word_index = sorted([self.get_index(w, word_dic) for w in merge_word])

			reserve_index = merge_word_index[-1]

			temp_word = word_dic[reserve_index]

			new_word = ''
			for j in merge_word_index:
				new_word += word_dic[j]

			# 修改句法关系
			for i in merge_word_index:
				word_dic_i = word_dic[i]
				for key in arcs_dic:
					arcs_dic_key = arcs_dic[key]
					if word_dic_i in arcs_dic[key]:
						temp_dic = self.list_to_dic(arcs_dic_key)
						word_dic_i_index = self.get_index(word_dic_i, temp_dic)
						arcs_dic_key[word_dic_i_index] = new_word

			# 修改词性
			merge_word_pos = list(set([pos_dic[j] for j in merge_word_index]))

			if len(merge_word_pos) == 1:
				if 'num' in merge_word_pos:
					pos_dic[reserve_index] = 'num'
				elif 'n' in merge_word_pos:
					pos_dic[reserve_index] = 'n'
				else:
					pos_dic[reserve_index] = 'v'
			else:
				if 'indiv' in merge_word_pos:
					pos_dic[reserve_index] = 'indiv'
				elif 'num' in merge_word_pos:
					pos_dic[reserve_index] = 'num'
				elif 'n' in merge_word_pos:
					pos_dic[reserve_index] = 'n'
				else:
					pos_dic[reserve_index] = 'v'

			# 修改词
			word_dic[reserve_index] = new_word

			merge_word_index.remove(reserve_index)
			for j in merge_word_index:
				word_dic[j] = ''
				pos_dic[j] = ''
				arcs_dic[j] = []

		return word_dic, pos_dic, arcs_dic

	def senten_simplify_3(self, senten_word, word_dic, pos_dic, arcs_dic):
		"""
		句式精简第三步
		:param senten_word: 原始分词结果
		:param word_dic: 分词字典式结果
		:param pos_dic: 词性标注字典式结果
		:param arcs_dic: 句法分析字典式结果
		:return: word_dic, pos_dic, arcs_dic
		"""
		word_list = [word_dic[i] for i in word_dic if word_dic[i] != '']
		num = len(word_list)

		if num > 3:
			unchange_word_index = []
			for key in word_dic:
				if (senten_word[key] == word_dic[key]):
					unchange_word_index.append(key)

			relate_word = []
			for key in pos_dic:
				if pos_dic[key] == 'indiv':
					indiv_word = word_dic[key]
					indiv_index = key
					relate_word.append(arcs_dic[key][2])
				if pos_dic[key] == 'hed':
					hed_word = word_dic[key]

			for key in arcs_dic:
				if (indiv_word in arcs_dic[key]) and (key != indiv_index):
					relate_word.append(word_dic[key])

			relate_word = set(relate_word)
			# relate_word.remove(indiv_word)

			relate_word_index = [self.get_index(word, word_dic) for word in relate_word]

			for indiv_relate_word_index in relate_word_index:

				indiv_relate_word = word_dic[indiv_relate_word_index]
				if (indiv_relate_word_index in unchange_word_index) and \
					(pos_dic[indiv_relate_word_index] != 'num') and \
					(pos_dic[indiv_relate_word_index] != 'n') and \
					(pos_dic[indiv_relate_word_index] != 'hed'):
					if (hed_word not in arcs_dic[indiv_relate_word_index]) or \
						(indiv_relate_word in arcs_dic[indiv_index]):

						if indiv_relate_word_index > indiv_index:
							new_word = indiv_word + indiv_relate_word
						else:
							new_word = indiv_relate_word + indiv_word

						# 修改句法关系
						for key in arcs_dic:
							if indiv_relate_word in arcs_dic[key]:
								temp_dic = self.list_to_dic(arcs_dic[key])
								relate_word_index = self.get_index(indiv_relate_word, temp_dic)
								arcs_dic[key][relate_word_index] = new_word
							if indiv_word in arcs_dic[key]:
								temp_dic = self.list_to_dic(arcs_dic[key])
								indiv_word_index = self.get_index(indiv_word, temp_dic)
								arcs_dic[key][indiv_word_index] = new_word

						# 修改词
						word_dic[indiv_index] = new_word
						pos_dic[indiv_index] = 'indiv'

						indiv_word = new_word

						word_dic[indiv_relate_word_index] = ''
						pos_dic[indiv_relate_word_index] = ''
						arcs_dic[indiv_relate_word_index] = []

			return word_dic, pos_dic, arcs_dic
		else:
			return word_dic, pos_dic, arcs_dic

	def senten_simplify_norm(self, sentence):
		"""
		对语句进行句式精简
		:param sentnece: 待处理语句
		:return: 简化后语句的词语列表、词性列表
		"""
		# 语句分析初始化
		SP = self.SP
		# 获取语句分析结果
		senten_word, senten_pos, senten_arcs = SP.sentence_parse(sentence)

		# 对sentence中的个体词和功能词进行词性替换
		RP_Dic = self.ReplacePos_Dic
		for i in range(len(senten_word)):
			senten_word_i = senten_word[i]
			if senten_word_i in RP_Dic:
				senten_pos[i] = RP_Dic[senten_word_i]

		Logger.log_DEBUG.info('个体词和功能词词性标签替换完成！')

		# 对分词结果字典化
		sentenWord_dic = self.list_to_dic(senten_word)

		# 对词性标注结果字典化
		sentenPos_dic = self.list_to_dic(senten_pos)

		# 对依存分析结果字典化
		sentenArcs_dic = self.list_to_dic(senten_arcs)

		# 对原始字典进行复制，使用深拷贝
		new_sentenWordDic = copy.deepcopy(sentenWord_dic)
		new_sentenPosDic = copy.deepcopy(sentenPos_dic)
		new_sentenArcsDic = copy.deepcopy(sentenArcs_dic)

		# 标问句式精简过程
		Logger.log_DEBUG.info('====开始句式精简====')

		Logger.log_DEBUG.info('标问核心词确定')
		word_dic, pos_dic, arcs_dic = self.hed_verb_norm(new_sentenWordDic, new_sentenPosDic, new_sentenArcsDic)

		Logger.log_DEBUG.info('标问个体词确定')
		word_dic, pos_dic, arcs_dic = self.merge_indiv_norm(word_dic, pos_dic, arcs_dic)

		Logger.log_DEBUG.info('标问句式精简第一步：除个体词和核心动词以外的词进行合并')
		to_merge_index = [key for key in new_sentenPosDic if new_sentenPosDic[key] != 'indiv' and new_sentenPosDic[key] != 'hed']
		# print(to_merge_index)
		word_dic, pos_dic, arcs_dic = self.senten_simplify_1(to_merge_index, word_dic, pos_dic, arcs_dic)

		Logger.log_DEBUG.info('标问句式精简第二步：以核心动词为媒介，满足精简规则的词进行合并')
		word_dic, pos_dic, arcs_dic = self.senten_simplify_2(word_dic, pos_dic, arcs_dic)

		Logger.log_DEBUG.info('标问句式精简第三步：与个体词存在关系，且是未经过合并的词语，与个体词合并为新的个体词')
		word_dic, pos_dic, arcs_dic = self.senten_simplify_3(senten_word, word_dic, pos_dic, arcs_dic)

		for key in pos_dic:
			if pos_dic[key] == 'b':
				pos_dic[key] = 'n'
			# if pos_dic[key] == 'j':
			# 	pos_dic[key] = 'n'

		return word_dic, pos_dic, arcs_dic

	def template_fill(self, word_dic, pos_dic, model):
		"""
		:param word_dic: 分词字典式结果
		:param pos_dic: 词性标注字典式结果
		:param model: 复述模板
		:return: 复述结果
		"""
		model_word_str = model[0]
		model_pos_str = model[1]

		model_word_list = [i.replace(' ', '') for i in model_word_str.split(' ') if len(i) != 0]
		model_pos_list = [i.replace(' ', '') for i in model_pos_str.split(' ') if len(i) != 0]

		model_word_dic = self.list_to_dic(model_word_list)
		model_pos_dic = self.list_to_dic(model_pos_list)

		model_noun_index = [key for key in model_pos_dic if model_pos_dic[key] == 'n']

		noun_list = []
		for key in pos_dic:
			if pos_dic[key] == 'indiv':
				indiv_word = word_dic[key]
			if pos_dic[key] == 'hed':
				hed_word = word_dic[key]
			if pos_dic[key] == 'num':
				num_word = word_dic[key]
			if pos_dic[key] == 'n':
				noun_list.append(word_dic[key])

		# 个体词和核心动词替换
		for key in model_pos_dic:
			if model_pos_dic[key] == 'indiv':
				model_word_dic[key] = indiv_word
			if model_pos_dic[key] == 'hed':
				model_word_dic[key] = hed_word
			if model_pos_dic[key] == 'num':
				model_word_dic[key] = num_word

		# 名词替换
		num1 = len(model_noun_index)
		num2 = len(noun_list)
		senten_repeat_dic = []

		if num1 >= num2:
			temp_list = [list(i) for i in permutations(model_noun_index, num2)]
			for l in temp_list:
				copy_dic = copy.deepcopy(model_word_dic)
				temp_dic = dict(zip(l, noun_list))
				for key in temp_dic:
					copy_dic[key] = temp_dic[key]
				senten_repeat_dic.append(copy_dic)
		else:
			temp_list = [list(i) for i in permutations(noun_list, num1)]
			for l in temp_list:
				copy_dic = copy.deepcopy(model_word_dic)
				temp_dic = dict(zip(model_noun_index, l))
				for key in temp_dic:
					copy_dic[key] = temp_dic[key]
				senten_repeat_dic.append(copy_dic)

		return senten_repeat_dic

	def get_senten_repeat(self, sentence):
		"""
		获取语句的复述模板，并进行模板填充，即复述语句生成
		:param sentence: 待复述语句
		:return: 复述结果
		"""

		word_dic, pos_dic, _ = self.senten_simplify_norm(sentence)

		senten_pos_list = [pos_dic[key] for key in pos_dic if len(pos_dic[key]) != 0]
		senten_pos_str = ' '.join(senten_pos_list)

		senten_template = []
		if senten_pos_str in self.Init_Dic.Template_Dic:
			senten_template = self.Init_Dic.Template_Dic[senten_pos_str]

		# 获得所有复述模板
		result_dic = []
		for model in senten_template:
			temp_result = self.template_fill(word_dic, pos_dic, model)
			result_dic.extend(temp_result)

		# 获得所有复述结果
		result = []
		for i in range(len(result_dic)):
			temp_str = ''
			temp_dic = result_dic[i]
			for key in temp_dic:
				temp_str += temp_dic[key]

			result.append(temp_str)
		Logger.log_DEBUG.info(sentence)
		Logger.log_DEBUG.info('====复述完成====')

		result = list(set(result)) # 结果去重

		return result


if __name__ == '__main__':
	SR = Senten_Repeat()

	# sentence  = '贷记卡未出账单明细查询'
	# sentence  = '贷记卡小额免密管理'
	sentence  = '借记卡修改短信通知号码'
	# sentence = '贷记卡设备卡挂失'
	# sentence  = '抢修漏气'
	# sentence  = '借记卡重置查询密码'
	# sentence  = '代缴费业务缴费通讯费'
	# sentence  = '借记卡查询活期余额'
	# sentence  = '查询公建拆改费用'
	# sentence  = '查询公建更名资料'
	# sentence  = '查询公建燃气费收费标准'

	word_dic, pos_dic, arcs_dic = SR.senten_simplify_norm(sentence)
	print(word_dic)
	print(pos_dic)
	print(arcs_dic)

	# word_list, pos_list, arc_list = SR.SP.sentence_parse(sentence)
	# RP_Dic = SR.ReplacePos_Dic
	#
	# for i in range(len(word_list)):
	# 	senten_word_i = word_list[i]
	# 	if senten_word_i in RP_Dic:
	# 		pos_list[i] = RP_Dic[senten_word_i]
	#
	# print('==== 语句分析结果 ====')
	# print(word_list)
	# print(pos_list)
	# print(arc_list)
	# print()
	#
	# word_dic = SR.list_to_dic(word_list)
	# pos_dic = SR.list_to_dic(pos_list)
	# arc_dic = SR.list_to_dic(arc_list)
	#
	# word_1, pos_1, arc_1 = SR.hed_verb_norm(word_dic, pos_dic, arc_dic)
	# print('==== 核心动词确定后结果 ====')
	# print(word_1)
	# print(pos_1)
	# print(arc_1)
	# print()
	#
	# word_2, pos_2, arc_2 = SR.merge_indiv_norm(word_1, pos_1, arc_1)
	# print('==== 个体词确定后结果 ====')
	# print(word_2)
	# print(pos_2)
	# print(arc_2)
	# print()
	#
	# the_list = [2,3,4]
	# word_3, pos_3, arc_3 = SR.senten_simplify_1(the_list, word_2, pos_2, arc_2)
	# print('==== 句式精简第一步结果 ====')
	# print(word_3)
	# print(pos_3)
	# print(arc_3)
	#
	# word_4, pos_4, arc_4 = SR.senten_simplify_2(word_3, pos_3, arc_3)
	# print('==== 句式精简第二步结果 ====')
	# print(word_4)
	# print(pos_4)
	# print(arc_4)
	#
	# word_5, pos_5, arc_5 = SR.senten_simplify_3(word_list, word_4, pos_4, arc_4)
	# print('==== 句式精简第三步结果 ====')
	# print(word_5)
	# print(pos_5)
	# print(arc_5)

