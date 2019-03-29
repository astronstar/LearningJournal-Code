# coding:utf-8

import os
pwd = os.getcwd()

import sys
sys.path.append(pwd)

import pandas as pd
import xlrd
import xlsxwriter
from TemplateMatching.SentenceRepeat import Senten_Repeat
from TemplateMatching.SentenceParse import Senten_Parse

# 复述类初始化
SR = Senten_Repeat()

# 句式精简
def senten_tm(sentence):
	_, pos_dic, _ = SR.senten_simplify_norm(sentence)
	if len(pos_dic) != 0:
		senten_pos = [i for i in pos_dic.values() if len(i) > 0]

		result = ' '.join(senten_pos)
	else:
		result = ''

	return result

if __name__ == '__main__':
	# data = xlrd.open_workbook('./data/ynnx_senten.xlsx')
	data = xlrd.open_workbook('./data/hr_senten.xlsx')
	table = data.sheets()[0]  # 按sheet索引号读取数据
	nrows = table.nrows  # 所有行数

	# write_file = xlsxwriter.Workbook('./data/ynnx_seten_0912.xlsx')
	write_file = xlsxwriter.Workbook('./data/hr_seten_0912.xlsx')
	excel_sheet1 = write_file.add_worksheet('result')
	excel_sheet2 = write_file.add_worksheet('no_result')

	m, n = 0, 0
	for rownum in range(nrows):
		senten = table.row_values(rownum)[0]
		try:
			senten_word, senten_posdic, _ = SR.senten_simplify_norm(senten)
			senten_pos = ' '.join([i for i in senten_posdic.values() if i != ''])
			_, _, arcs_list = SR.SP.sentence_parse(senten)
			senten_arc = [' '.join(l) for l in arcs_list]

			senten_lst = [i for i in senten_word.values() if i != '']

			excel_sheet1.write(m, 0, senten)
			excel_sheet1.write(m, 1, ' '.join(senten_lst))
			excel_sheet1.write(m, 2, senten_pos)
			excel_sheet1.write(m, 3, ', '.join(senten_arc))

			m += 1
		except Exception as e:

			words_list, pos_list, arcs_list = SR.SP.sentence_parse(senten)
			senten_arc = [' '.join(l) for l in arcs_list]

			excel_sheet2.write(n, 0, senten)
			excel_sheet2.write(n, 1, ' '.join(words_list))
			excel_sheet2.write(n, 2, ' '.join(pos_list))
			excel_sheet2.write(n, 3,  ', '.join(senten_arc))
			n += 1
			continue

	write_file.close()