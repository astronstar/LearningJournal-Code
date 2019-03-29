# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年7月10日
    文件名：	InitializeDict.py
    功能：	初始化词典，包含
    			- 个体词-标签词典
    			- 功能词-标签词典
    			- 标签-功能词列表词典
    			- 复述模板词典
"""
import os
pwd = os.getcwd() # 获取当前路径
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.') # dirname返回pwd的目录，abspath返回规范化的绝对路径

import sys
sys.path.append(father_path) # 将father_path路径加到当前模块扫描的路径里

import config_TM
from TemplateMatching.MyLog import Logger
import collections


class Initial_Dict_Load:
	def __init__(self, IndivWordDict={}, WordTagDict={}, TagWordDict={}, TemplateDict={}, indivword_path='', featureword_path='', template_path=''):
		"""
		初始化操作
		:param IndivWordDict: 个体词-标签词典
		:param WordTagDict: 功能词-标签词典
		:param TagWordDict: 标签-功能词列表词典
		:param TemplateDict: 复述模板词典
		:param indivword_path: 个体词读取路径
		:param featureword_path: 功能词读取路径
		:param template_path: 复述模板读取路径
		"""
		# 个体词-标签词典(key:个体词, value:标签)
		self.IndivWord_Dic = IndivWordDict

		# 功能词-标签词典(key:功能词, value:标签)
		self.WordTag_Dic = WordTagDict

		# 词性-功能词字典(key:标签, value:功能词列表)
		self.TagWord_Dic = TagWordDict

		# 复述模板字典
		TemplateDict = collections.defaultdict(list)
		self.Template_Dic = TemplateDict

		# 个体词读取路径
		self.indivword_path = indivword_path

		# 功能词读取路径
		self.featureword_path = featureword_path

		# 复述模板读取路径
		self.template_path = template_path

		# 个体词-标签词典初始化
		if indivword_path != "":
			self.Init_IndivWord_Dic(self.IndivWord_Dic, self.indivword_path)
			Logger.log_DEBUG.info("个体词-标签词典初始化完成！词典大小%d" % len(self.IndivWord_Dic))
		else:
			Logger.log_ERROR.error("没有个体词文件！")

		# 功能词-标签词典、标签-功能词列表词典初始化
		if featureword_path != "":
			self.Init_WordTag_Dic(self.WordTag_Dic, self.TagWord_Dic, self.featureword_path)
			Logger.log_DEBUG.info("功能词-标签词典初始化完成！词典大小%d" % len(self.WordTag_Dic))
			Logger.log_DEBUG.info("标签-功能词列表词典初始化完成！词典大小%d" % len(self.TagWord_Dic))
		else:
			Logger.log_ERROR.error("没有功能词文件！")

		# 复述模板词典初始化
		if template_path != "":
			self.Init_Template_Dic(self.Template_Dic, self.template_path)
			Logger.log_DEBUG.info("复述模板词典初始化完成！词典大小%d" % len(self.Template_Dic))
		else:
			Logger.log_ERROR.error("没有复述模板文件！")

	def Init_IndivWord_Dic(self, IndivWordDict, indivword_path):
		"""
		个体词-标签词典初始化
		:param indivword_path: 个体词读取路径
		:return: 返回个体词-标签词典
		"""
		try:
			if os.path.exists(indivword_path):
				fr = open(indivword_path, 'r', encoding='utf8')
				for line in fr.readlines():
					if line:
						line_temp = [w.replace('\n', '').replace('\r', '').replace('\t', '') for w in line.split(' ') if
						             len(w) != 0]
						if len(line_temp) > 1:
							IndivWordDict[line_temp[0]] = line_temp[1]
			else:
				Logger.log_ERROR.error('找不到个体词文件，请检查传入路径是否正确！')
		except Exception as e:
			s = '初始化词典发生异常Init_IndivWord_Dic' + str(e)
			Logger.log_ERROR.error(s)

	def Init_WordTag_Dic(self, WordTagDict, TagWordDict, featureword_path):
		"""
		功能词-标签和标签-功能词列表词典初始化
		:param WordTagDict: 功能词-标签词典
		:param TagWordDict: 标签-功能词列表词典
		:param featureword_path: 功能词读取路径
		:return: 返回功能词-标签词典和标签-功能词列表词典
		"""
		try:
			if os.path.exists(featureword_path):
				fr = open(featureword_path, 'r', encoding='utf8')
				# WordTagDict, TagWordDict = {}, {}
				for line in fr.readlines():
					if line:
						line_temp = [w.replace('\n', '').replace('\r', '').replace('\t', '') for w in line.split(' ') if
						             len(w) != 0]
						if len(line_temp) > 1:
							key_word_list = line_temp[:-1]
							key_value = line_temp[-1]
							TagWordDict[line_temp[-1]] = line_temp[:-1]

						for key_word in key_word_list:
							WordTagDict[key_word] = key_value
			else:
				Logger.log_ERROR.error('找不到功能词文件，请检查传入路径是否正确！')
		except Exception as e:
			s = '初始化词典发生异常Init_WordTag_Dic' + str(e)
			Logger.log_ERROR.error(s)

	def Init_Template_Dic(self, TemplateDict, template_path):
		"""
		复述模板词典初始化
		:param TemplateDict: 复述模板库词典
		:param template_path: 复述模板库读取路径
		:return: 返回复述模板词典
		"""
		try:
			if os.path.exists(template_path):
				fr = open(template_path, 'r', encoding='utf8')
				for line in fr.readlines():
					if line:
						line_temp = [w.replace('\n', '').replace('\r', '').replace('\t', '') for w in line.split(',') if
						             len(w) != 0]
						if len(line_temp) > 1:
							TemplateDict[line_temp[0]].append(line_temp[1:])
			else:
				Logger.log_ERROR.error('找不到复述模板文件，请检查传入路径是否正确！')
		except Exception as e:
			s = '初始化词典发生异常Init_Template_Dic' + str(e)
			Logger.log_ERROR.error(s)


if __name__ == '__main__':
	InitDict = Initial_Dict_Load({}, {}, {}, {}, config_TM.indivword_path, config_TM.featureword_path,
	                             config_TM.template_path)

	print(InitDict.WordTag_Dic)
	print(InitDict.TagWord_Dic)
	print(InitDict.Template_Dic)