# coding:utf-8

"""
	作者：	仝哲
	日期：	2018年10月22日
	文件名：	config.py
	功能：	配置文件
"""

import configparser
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
# 日志保存路径
log_path = FILE_PATH + '/log/'
if not os.path.exists(log_path):
	os.makedirs(log_path)
	
info_file = os.path.join(log_path, 'info_log')  # 非error信息日志记录文件
err_file = os.path.join(log_path, 'err_log')  # error信息日志记录文件