# coding:utf-8

"""
	作者：	仝哲
	日期：	2018年10月22日
	文件名：	IdentifyWeb.py
	功能：	多标签分类识别功能web服务
"""

import GruModel
import TextcnnModel
import MLModel
import tornado.ioloop
from tornado.wsgi import WSGIContainer
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import tornado.web
import datetime
import traceback
from MyLog import Logger


class MainHandler(tornado.web.RequestHandler):
	executor = ThreadPoolExecutor(1)
	@run_on_executor
	def get(self):
		self.write("Hello, world!")

	@run_on_executor
	def post(self):
		starttime = datetime.datetime.now()

		item_id = str(self.get_argument('item_id', '')) # 项目id
		model_id = str(self.get_argument('model_id', ''))

		ip = str(self.get_argument('ip', ''))
		up_url = str(self.get_argument('up_url', ''))
		down_url = str(self.get_argument('down_url', ''))
		access_url = str(self.get_argument('access_url', ''))
		access_key = str(self.get_argument('access_key', ''))
		_init_companyId = str(self.get_argument('_init_companyId', ''))

		model_type = str(self.get_argument('model_type', '')) # 模型类型
		data_id = str(self.get_argument('data_id', '')) # 数据id

		Logger.log_DEBUG.info("==== 识别接口url获取参数打印 ====")
		Logger.log_DEBUG.info("item_id: %s" % item_id)
		Logger.log_DEBUG.info("model_id: %s" % model_id)
		
		Logger.log_DEBUG.info("ip: %s" % ip)
		Logger.log_DEBUG.info("up_url: %s" % up_url)
		Logger.log_DEBUG.info("down_url: %s" % down_url)
		Logger.log_DEBUG.info("access_url: %s" % access_url)
		Logger.log_DEBUG.info("access_key: %s" % access_key)
		Logger.log_DEBUG.info("_init_companyId: %s" % _init_companyId)
		Logger.log_DEBUG.info("model_type: %s" % model_type)
		Logger.log_DEBUG.info("data_id: %s" % data_id)

		if model_type == "":
			model_type = "AD_BR"

		try:
			if model_type == 'GRU':
				gru_mlb_file_id = str(self.get_argument('mlb_file_id', '')) # mlb模型id
				gru_tokenizer_file_id = str(self.get_argument('tokenizer_file_id', '')) # tokenizer模型id
				gru_model_file_id = str(self.get_argument('model_file_id', '')) # 模型id
				max_sequence_length = str(self.get_argument('max_sequence_length', '')) # 最大词个数
				batch_size = str(self.get_argument('batch_size', '')) # batch大小

				if max_sequence_length == "":
					max_sequence_length = "5000"
				if batch_size == "":
					batch_size = "128"

				print("Start GRU Predict")
				gru_result = GruModel.gru_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, gru_mlb_file_id, gru_tokenizer_file_id, gru_model_file_id, max_sequence_length, batch_size)
				print("End GRU Predict")
				self.write(gru_result)

			elif model_type == 'TEXTCNN':
				textcnn_mlb_file_id = str(self.get_argument('mlb_file_id', '')) # mlb模型id
				textcnn_tokenizer_file_id = str(self.get_argument('tokenizer_file_id', '')) # tokenizer模型id
				textcnn_model_file_id = str(self.get_argument('model_file_id', '')) # 模型id
				max_sequence_length = str(self.get_argument('max_sequence_length', '')) # 最大词个数
				batch_size = str(self.get_argument('batch_size', '')) # batch大小

				if max_sequence_length == "":
					max_sequence_length = "5000"
				if batch_size == "":
					batch_size = "128"

				print("Start TEXTCNN Predict")
				textcnn_result = TextcnnModel.textcnn_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, textcnn_mlb_file_id, textcnn_tokenizer_file_id, textcnn_model_file_id, max_sequence_length, batch_size)
				print("End TEXTCNN Predict")
				self.write(textcnn_result)

			elif model_type == 'MLKNN':
				knn_tfidf_file_id = str(self.get_argument('tfidf_file_id', ''))
				knn_mlb_file_id = str(self.get_argument('mlb_file_id', ''))
				knn_model_file_id = str(self.get_argument('model_file_id', ''))

				print('Start MLKNN Predict')
				ml_result = MLModel.knn_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, knn_tfidf_file_id, knn_mlb_file_id, knn_model_file_id)
				print('End MLKNN Predict')
				self.write(ml_result)

			elif model_type == 'AD_BR':
				br_tfidf_file_id = str(self.get_argument('tfidf_file_id', ''))
				br_mlb_file_id = str(self.get_argument('mlb_file_id', ''))
				br_model_file_id = str(self.get_argument('model_file_id', ''))

				print('Start AD_BR Predict')
				br_result = MLModel.br_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, br_tfidf_file_id, br_mlb_file_id, br_model_file_id)
				print('End AD_BR Predict')
				self.write(br_result)

			elif model_type == 'AD_CC':
				cc_tfidf_file_id = str(self.get_argument('tfidf_file_id', ''))
				cc_mlb_file_id = str(self.get_argument('mlb_file_id', ''))
				cc_model_file_id = str(self.get_argument('model_file_id', ''))

				print('Start AD_CC Predict')
				cc_result = MLModel.cc_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, cc_tfidf_file_id, cc_mlb_file_id, cc_model_file_id)
				print('End AD_CC Predict')
				self.write(cc_result)

			elif model_type == 'AD_LP':
				lp_tfidf_file_id = str(self.get_argument('tfidf_file_id', ''))
				lp_mlb_file_id = str(self.get_argument('mlb_file_id', ''))
				lp_model_file_id = str(self.get_argument('model_file_id', ''))

				print('Start AD_LP Predict')
				lp_result = MLModel.lp_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, lp_tfidf_file_id, lp_mlb_file_id, lp_model_file_id)
				print('End AD_LP Predict')
				self.write(lp_result)
				
		except Exception as e:
			# print('请检查参数输入是否正确' + str(e))
			# print('==== 错误信息 ====')
			# print('traceback.print_exc():', traceback.print_exc())
			# print('========')
			Logger.log_ERROR.error("请检查参数输入是否正确：" + str(e))
			Logger.log_ERROR.error("错误详细信息：%s" % traceback.print_exc())
			# Logger.log_ERROR.error(traceback.print_exc())

		endtime = datetime.datetime.now()
		time_diff = endtime - starttime
		print('耗时:', time_diff)
		Logger.log_DEBUG.info("==== use time: %s" % str(time_diff))

if __name__ == '__main__':
	app = tornado.web.Application([(r"/", MainHandler), ])
	sockets = tornado.netutil.bind_sockets(9000)# 绑定端口创建套接字
	tornado.process.fork_processes(0)# 开启多进程，0表示自动识别CPU核心数
	server = tornado.httpserver.HTTPServer(app)# 创建http服务
	server.add_sockets(sockets)# 将套接字添加到http服务中
	tornado.ioloop.IOLoop.instance().start() # 启动io事件循环