# coding:utf-8

"""
    作者： 仝哲
    日期： 2018年10月22日
    文件名：    DataLoad.py
    功能： 文件服务器连接，文件服务器数据下载及DataFrame格式数据保存
"""

import requests
import json
import traceback
import os
import io
import pandas as pd
from MyLog import Logger


class Data_Load:
	"""
	文件服务器操作类
	"""
	def __init__(self, ip, up_url, down_url, access_url, access_key, _init_companyId):
		"""
		文件服务器操作初始化
		:param ip: ip地址
		:param up_url: 上传地址url
		:param down_url: 下载地址url
		:param access_url: 获取票据的url
		:param access_key: 获取票据的key
		:param _init_companyId:
		"""
		self.ip = "http://" + ip
		self.up_url = self.ip + up_url
		self.down_url = self.ip + down_url
		self.access_url = self.ip + access_url
		self.accessKey = access_key
		self._init_companyId = _init_companyId
		self.get_api_ticket()

	def get_api_ticket(self):
		try:
			url = self.access_url+"?accessKey="+self.accessKey
			# 票据
			r = requests.get(url)
			p = r.text#.encode('utf-8')
			# print('get_api_ticket : '+ p)
			ret = json.loads(p)
			self.api_ticket = ret['data']
			return self.api_ticket
		except:
			print ("Error--------------get accessToken,maybe some parameters are error or service is not on")
			print (traceback.format_exc())

	#通过文件内容上传文件服务器
	#file_data的数据类型
	def upload_file_by_data(self, file_data):
		files = {'file':file_data}
		#票据
		headers={'api-ticket':self.api_ticket,
				 '_init_companyId':self._init_companyId
				 }
		try:

			r = requests.post(self.up_url,headers=headers,files=files)
			p = r.text#.encode('utf-8')
			# print('upload file : ' + p)
			#将string转成dict
			ret_dict = json.loads(p)

			data = ret_dict['fileAttributes']['fileId']
			return data
		except :
			print ("Error--------------upload file to file-server error")
			print (traceback.format_exc())


	#根据id和类型判断取出的文件是什么内容的
	def download_file_by_id(self, id):
		try:
			url = self.down_url + "?fileId=" + id
			# 票据
			headers = {'api-ticket': self.api_ticket,
					   '_init_companyId': self._init_companyId
					   }
			r = requests.get(url,headers=headers)
			p = r.text
			return p
		except:
			print ("Error--------------down file from file-server error")
			print (traceback.format_exc())

	# 根据id
	def download_model_by_id(self, id):
		try:
			url = self.down_url + "?fileId=" + id
			# 票据
			headers = {'api-ticket': self.api_ticket,
					'_init_companyId': self._init_companyId
						}
			r = requests.get(url, headers=headers)

			if r.content == b'':
				return r.text
			else:
				return r.content
		except:
			print("Error--------------down file from file-server error")
			print(traceback.format_exc())

	#根据id
	def delete_file_by_id(self, id):
		try:
			url = self.delete_url
			# 票据
			headers = {'api-ticket': self.api_ticket,
					   '_init_companyId': self._init_companyId,
					   'Content-Type':'application/json'
					   }
			content = []
			content.append(id)
			# print(content)
			r = requests.post(url,headers=headers,json=content)
			# print('delete file : ' + r.text)
		except:
			print ("Error--------------delete file from file-server error")
			print (traceback.format_exc())

	def save_data(self, p):
		"""
		将数据保存为DataFrame格式
		:param p: 相关数据
		:return: DataFrame格式数据
		"""
		if p != None:
			TESTDATA = io.StringIO(p)

			df = pd.read_csv(TESTDATA, sep=',', encoding='utf-8', names=['id', 'context', 'label'])
			df['context'] = df['context'].apply(lambda x: x.replace('\n', ''))

			return df

		return None

	def save_predict_data(self, p):
		"""
		将测试数据保存为DataFrame格式
		:param p: 相关数据
		:return: DataFrame格式数据
		"""
		if p != None:
			TESTDATA = io.StringIO(p)

			df = pd.read_csv(TESTDATA, sep='##', encoding='utf-8', names=['id', 'context'])
			df['context'] = df['context'].apply(lambda x: x.replace('\n', ''))

			return df

		# return None

	def get_train_data(self, train_id):
		"""
		获取文件服务器上相关文件数据
		:param train_id: 文件服务器上该文件id
		:return: 格式为DataFrame的文件数据
		"""
		train_data = self.download_file_by_id(train_id)
		train_data = self.save_data(train_data)

		return train_data

if __name__ == '__main__':
	ip = '10.0.2.121:80'
	up_url = '/fileManager/api/file/uploadFileP'
	down_url = '/fileManager/api/file/downloadFileP'

	delete_url = '/fileManager/api/file/deleteFiles'

	access_url = '/api/gateway/ticket'
	access_key = '33a832d7949c11e89024000c2961e520'
	_init_companyId = '08d181119a7b4c0e94ff368942fd4420'

	train_id = "87ea99c4d11411e8be95000c2961e520"

	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)

	train_data = dl.get_train_data(train_id)

	print(type(train_data))
	print(train_data.head())

	# test = "123##今天是一个，好日子"
	# df = dl.save_predict_data(test)
	# print(df)