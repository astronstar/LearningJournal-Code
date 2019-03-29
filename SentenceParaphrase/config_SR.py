# coding:utf-8

"""
    作者：	钱艳
    日期：	2018年7月9日
    文件名：	config_SR.py
    功能：	同义词替换配置文件
"""
#哈工大同义词路径
#个体词同义词路径
import os
# # 配置文件路径
# path = os.getcwd().replace('\\', '/')
# print(path)
#print("获取当前文件路径——" + os.path.realpath(__file__))  # 获取当前文件路径
parent = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
print("获取其父目录——" + parent)
# 数据文件夹名
data_file = parent+'/data/data_SR/'
# 哈工大词林文件路径
sys_path = data_file + '/sysnonym_final.txt'

# 个体同义词文件路径
sys_indiv_path =data_file + '/synonym_indiv.txt'

# 自定义个体词典文件路径
userdict_path = data_file+'/indiv_dict.txt'

#语言模型判断最大数量
max_num=2000

# #同义词替换文件路径
# input_SR=parent+'/result/test_result_TZ_20180607.xlsx'
# #替换后输出结果文件路径
# output_SR=parent+'/result/result_final_QY.xlsx'

#标问路径
Questioning=parent+'/data/data_TM/hzrq_data.xlsx'
#标问是否需要复述
Questioning_flag=True

# 日志保存路径
log_path =  parent+'/log/log_SR/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
# 非error信息日志记录文件
info_file =log_path+'info_log.txt'
print(info_file)
# error信息日志记录文件
err_file = log_path+'err_log.txt'

#语言模型地址
url="http://10.0.10.116:3000/"

#语言模型返回句子个数
s_num=5