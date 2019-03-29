# coding:utf-8
"""
    作者：	钱艳
    日期：	2018年07月9日
    文件名：	Batch_SR.py
    功能：	批量读入文件得到同义词替换结果
"""

import xlrd
import xlsxwriter
from SynonymReplacement.Synonym_Replacement import Synonym_Replace
import config_SR


def result_output(file_read, file_out, sheet1_name, sheet2_name):
    '''
    复述结果写入到excel
    :param file_read: 待复述语句读取文件名
    :param file_out: 结果输出文件名
    :param sheet1_name: 第一个sheet名字(记录有复述语句的复述结果)
    :param sheet2_name: 第二个sheet名字(记录无复述语句)
    :return: 
    '''
    # 初始化话同义词替换类对象
    sys_rep = Synonym_Replace()
    sys_rep.init_sys_dic()
    # 待复述语句数据读取
    data = xlrd.open_workbook(file_read)
    table = data.sheets()[0]  # 按sheet索引号读取数据
    nrows = table.nrows  # 所有数据的行数
    # 打开复述语句存储文件
    write_file = xlsxwriter.Workbook(file_out)
    excel_sheet1 = write_file.add_worksheet(sheet1_name)
    excel_sheet2 = write_file.add_worksheet(sheet2_name)
    C = 1
    HW = 0  # 复述成功行数标记
    excel_sheet1.write(HW, 0, "sentence")  # 原句
    excel_sheet1.write(HW, 1, "senten_repeat")  # 复述结果
    excel_sheet1.write(HW, 2, "type")  # 标签号
    HW = HW + 1
    HWf = 0  # 复述失败行数标记
    excel_sheet2.write(HWf, 0, "sentence")  # 原句
    HWf = HWf + 1
    for H in range(1, nrows):
        s = table.cell(H, C).value
        type_L = table.cell(H, C + 1).value
        s_sys = sys_rep.Synonym_Replace_mian(s, config_SR.s_num)

        if len(s_sys) == 1:
            excel_sheet2.write(HWf, 0, s)  # 原句
            HWf = HWf + 1
        else:
            for r in s_sys:
                excel_sheet1.write(HW, 0, s)  # 原句
                excel_sheet1.write(HW, 1, r)  # 复述结果
                excel_sheet1.write(HW, 2, type_L)  # 标签号
                HW = HW + 1

    #判断标问是否需要复述
    if config_SR.Questioning_flag:
        # 标问数据读取
        data_Q = xlrd.open_workbook(config_SR.Questioning)
        table_Q = data_Q.sheets()[0]  # 按sheet索引号读取数据
        nrows_Q = table_Q.nrows  # 所有数据的行数
        C = 0
        for H in range(1, nrows_Q):
            s = table_Q.cell(H, C).value
            type_L = table_Q.cell(H, C + 1).value
            s_sys = sys_rep.Synonym_Replace_mian(s, config_SR.s_num)

            if len(s_sys) == 1:
                excel_sheet2.write(HWf, 0, s)  # 原句
                HWf = HWf + 1
            else:
                for r in s_sys:
                    excel_sheet1.write(HW, 0, s)  # 原句
                    excel_sheet1.write(HW, 1, r)  # 复述结果
                    excel_sheet1.write(HW, 2, type_L)  # 标签号
                    HW = HW + 1

    return 'Finished'

if __name__ == '__main__':
    fr = './result/result_hzrq_20180709.xlsx'
    fo = './result/test_result_QY.xlsx'
    sheet1_name = 'repeat_result'
    sheet2_name = 'no_template'

    result_output(fr, fo, sheet1_name, sheet2_name)