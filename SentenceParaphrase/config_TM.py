# coding:utf-8

"""
    作者：	仝哲
    日期：	2018年7月10日
    文件名：	config_TM.py
    功能：	配置文件，包含
    			- 日志保存路径
    			- LTP模型加载路径
    			- 自定义分词文件读取路径
    			- 个体词文件读取路径
    			- 功能词文件读取路径
    			- 复述模板文件读取路径
"""

import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

# 日志保存路径
log_path = FILE_PATH + '/log/log_TM'

if not os.path.exists(log_path):
	os.makedirs(log_path)

info_file = os.path.join(log_path, 'info_log')  # 非error信息日志记录文件
err_file = os.path.join(log_path, 'err_log')  # error信息日志记录文件

# ltp模型目录的路径
LTP_DATA_DIR = FILE_PATH + '/TemplateMatching/LTP/ltp_data'

# 分词模型路径，模型名称为 cws.model
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

# 词性标注模型路径，模型名称为 pos.model
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

# 依存句法分析模型路径，模型名称为 parser.model
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')

# 数据读取路径
dataset_path = FILE_PATH + '/data/data_TM'


# 自定义分词文件读取路径
userdict_path = os.path.join(dataset_path, 'userdict_ynnx.txt') # 云南农信
# userdict_path = os.path.join(dataset_path, 'userdict_hzrq.txt') # 杭燃

# 个体词文件读取路径
indivword_path = os.path.join(dataset_path, 'indivword_ynnx.txt') # 云南农信
# indivword_path = os.path.join(dataset_path, 'indivword_hzrq.txt') # 杭燃

# 功能词文件读取路径
featureword_path = os.path.join(dataset_path, 'featureword_ynnx.txt') # 云南农信
# featureword_path = os.path.join(dataset_path, 'featureword_hzrq.txt') # 杭燃

# 复述模板文件读取路径
template_path = os.path.join(dataset_path, 'template_ynnx.txt') # 云南农信
# template_path = os.path.join(dataset_path, 'template_hzrq.txt') # 杭燃
