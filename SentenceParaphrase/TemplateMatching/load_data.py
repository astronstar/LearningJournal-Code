# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年9月17日
    文件名：	IndivWord_Replace.py
    功能：	停用词表初始化
"""

import os
pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')

import sys
sys.path.append(father_path)

import collections

import config_TM
from TemplateMatching.MyLog import Logger

class LoadData:

	def __init__(self, stopword=[], senten={}, stopword_path='', senten_path=''):
		self.stopword = stopword

		senten = collections.defaultdict(list)
		self.senten = senten

		if stopword_path != '':
			self.load_stopword(stopword, stopword_path)
			Logger.log_DEBUG.info("停用词初始化完成！大小为：%d" % len(self.stopword))

		if senten_path != '':
			self.load_senten(senten, senten_path)
			Logger.log_DEBUG.info('语句初始化完成！大小为：%d' % len(self.senten))

	def load_stopword(self, stopword, stopword_path):
		try:
			if os.path.exists(stopword_path):
				fr = open(stopword_path, 'r', encoding='utf-8')
				for line in fr.readlines():
					line = line.strip().replace('\n', '').replace(' ', '')
					stopword.append(line)
			else:
				Logger.log_ERROR.error('找不到停用词文件，请检查传入路径是否正确！')
		except Exception as e:
			s = '初始化停用词发生异常load_stopword' + str(e)
			Logger.log_ERROR.error(s)

	def load_senten(self, senten, senten_path):
		try:
			if os.path.exists(senten_path):
				fr = open(senten_path, 'r', encoding='utf-8')
				for line in fr.readlines():
					if line:
						temp = [s.strip().replace('\n', '').replace(' ', '') for s in line.split(',')]
						# print(temp)
						senten[temp[0]].append(temp[1])
			else:
				Logger.log_ERROR.error('找不到语句文件，请检查传入路径是否正确！')
		except Exception as e:
			s = '初始化语句文件发生异常load_senten' + str(e)
			Logger.log_ERROR.error(s)


if __name__ == '__main__':
    LD = LoadData(stopword=[], senten={}, stopword_path='E:/stopword.txt', senten_path='E:/test.txt')
























