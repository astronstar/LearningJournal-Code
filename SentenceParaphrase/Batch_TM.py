# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年7月10日
    文件名：	Batch_TM.py
    功能：	复述结果写入到excel中
"""

import os
pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')

import sys
sys.path.append(pwd)

import xlrd
import xlsxwriter
from TemplateMatching.SentenceRepeat import Senten_Repeat
from TemplateMatching.MyLog import Logger


def cal_sum(the_list):
	"""
	计算列表前n项和
	:param the_list: 待处理列表
	:return: 列表求和结果，默认第一项为0
	"""
	list_len = [len(l) for l in the_list]

	list_sum = []

	num = 0
	list_sum.append(num)
	for i in range(len(list_len)):
		num += list_len[i]
		list_sum.append(num)

	return list_sum


def result_output(file_read, file_out, sheet1_name, sheet2_name):
	"""
	复述结果写入到excel
	:param file_read: 待复述语句读取文件名
	:param file_out: 结果输出文件名
	:param sheet1_name: 第一个sheet名字(记录有复述语句的复述结果)
	:param sheet2_name: 第二个sheet名字(记录无复述语句)
	:return: 打印Finished
	"""
	SR = Senten_Repeat()

	# 待复述语句数据读取
	data = xlrd.open_workbook(file_read)
	table = data.sheets()[0]  # 按sheet索引号读取数据
	nrows = table.nrows # 所有行数

	write_file = xlsxwriter.Workbook(file_out)
	excel_sheet1 = write_file.add_worksheet(sheet1_name)
	excel_sheet2 = write_file.add_worksheet(sheet2_name)

	all_row = []
	all_row_id = []
	result_list = []
	no_template = []
	no_template_id = []

	for rownum in range(1, nrows):
		row = table.row_values(rownum)

		try:
			temp_list = SR.get_senten_repeat(row[0])
			temp_list = list(set(temp_list))
			if len(temp_list) != 0:
				all_row.append(row[0])
				all_row_id.append(row[1])
				result_list.append(temp_list)
			else:
				no_template.append(row[0])
				no_template_id.append(row[1])
		except Exception as e:
			no_template.append(row[0])
			no_template_id.append(row[1])
			s = str(e)
			Logger.log_ERROR.error(s)
			# continue

	no_template_dic = dict(zip(no_template, no_template_id))

	excel_sheet2.write(0,0,'sentence')
	excel_sheet2.write(0,1,'type')

	for i in range(len(no_template_dic)):
		excel_sheet2.write(i+1, 0, no_template[i])
		excel_sheet2.write(i+1, 1, no_template_id[i])


	all_row_tuple = list(zip(all_row, all_row_id))
	all_row_dic = dict((i, c) for i, c in enumerate(all_row_tuple))

	result = [i for _ in result_list for i in _]
	result_dic = dict((i, c) for i, c in enumerate(result))

	result_sum = cal_sum(result_list)

	# 复述结果excel写入
	excel_sheet1.write(0, 0, 'sentence')
	excel_sheet1.write(0, 1, 'senten_repeat')
	excel_sheet1.write(0, 2, 'type')
	for key in all_row_dic:
		senten = all_row_dic[key][0]
		senten_id = all_row_dic[key][1]

		low = result_sum[key]
		up = result_sum[key + 1]

		for i in range(low, up):
			excel_sheet1.write(i+1, 0, senten) # 原句
			excel_sheet1.write(i+1, 1, result_dic[i]) # 复述结果
			excel_sheet1.write(i+1, 2, senten_id) # 标签号

	write_file.close()

	return 'Finished'


if __name__ == '__main__':
	# fr = './data/data_TM/test_ynnx_3.xlsx'
	# fo = './result/result_ynnx.xlsx'

	fr = './data/data_TM/test_all_70.xlsx'
	fo = './result/result_ynnx.xlsx'

	sheet1_name = 'repeat_result'
	sheet2_name = 'no_template'

	result_output(fr, fo, sheet1_name, sheet2_name)
