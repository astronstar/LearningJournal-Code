# coding:utf-8

"""
	作者：	仝哲
	日期：	2018年10月22日
	文件名：	TrainWeb.py
	功能：	多标签分类模型训练功能web服务
"""

import GruModel
import TextcnnModel
import MLModel
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

		Logger.log_DEBUG.info("==== 训练接口url获取参数打印 ====")
		Logger.log_DEBUG.info("item_id: %s" % item_id)
		Logger.log_DEBUG.info("model_id: %s" % model_id)
		
		Logger.log_DEBUG.info("ip: %s" % ip)
		Logger.log_DEBUG.info("up_url: %s" % up_url)
		Logger.log_DEBUG.info("down_url: %s" % down_url)
		Logger.log_DEBUG.info("access_url: %s" % access_url)
		Logger.log_DEBUG.info("access_key: %s" % access_key)
		Logger.log_DEBUG.info("_init_companyId: %s" % _init_companyId)
		Logger.log_DEBUG.info("model_type: %s" % model_type)

		if model_type == "":
			model_type = "AD_BR"
		train_data_id = str(self.get_argument('train_data_id', '')) # 数据id

		Logger.log_DEBUG.info("train_data_id: %s" % train_data_id)

		try:
			if model_type == 'GRU':
				w2v_size = str(self.get_argument('w2v_size', '')) # 词向量维度
				w2v_window = str(self.get_argument('w2v_window', ''))
				w2v_min_count = str(self.get_argument('w2v_min_count', ''))
				w2v_negative = str(self.get_argument('w2v_negative', ''))
				batch_size = str(self.get_argument('batch_size', '')) # batch大小
				epochs = str(self.get_argument('epochs', '')) # 迭代次数
				max_sequence_length = str(self.get_argument('max_sequence_length', '')) # 最大词个数
				num_filter = str(self.get_argument('num_filter', '')) # 过滤器个数
				drop_rate = str(self.get_argument('drop_rate', '')) # 衰减率

				if w2v_size == "":
					w2v_size = "300"
				if w2v_window == "":
					w2v_window = "5"
				if w2v_min_count == "":
					w2v_min_count = "1"
				if w2v_negative == "":
					w2v_negative = "5"
				if batch_size == "":
					batch_size = "128"
				if epochs == "":
					epochs = "40"
				if max_sequence_length == "":
					max_sequence_length = "5000"
				if num_filter == "":
					num_filter = "128"
				if drop_rate == "":
					drop_rate = "0.4"

				print("Start GRU Training")
				gru_result = GruModel.gru_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, w2v_size, w2v_window, w2v_min_count, w2v_negative, batch_size, epochs, max_sequence_length, num_filter, drop_rate)
				print("End GRU Training")
				self.write(gru_result)
				# self.write('mlb_id:', gru_mlb_id)
				# self.write('tokenizer_id:', gru_tokenizer_id)
				# self.write('model_id:', gru_model_id)

			elif model_type == 'TEXTCNN':
				w2v_size = str(self.get_argument('w2v_size', '')) # 词向量维度
				w2v_window = str(self.get_argument('w2v_window', ''))
				w2v_min_count = str(self.get_argument('w2v_min_count', ''))
				w2v_negative = str(self.get_argument('w2v_negative', ''))
				batch_size = str(self.get_argument('batch_size', '')) # batch大小
				epochs = str(self.get_argument('epochs', '')) # 迭代次数
				max_sequence_length = str(self.get_argument('max_sequence_length', '')) # 最大词个数
				num_filter = str(self.get_argument('num_filter', '')) # 过滤器个数
				drop_rate = str(self.get_argument('drop_rate', '')) # 衰减率

				if w2v_size == "":
					w2v_size = "300"
				if w2v_window == "":
					w2v_window = "5"
				if w2v_min_count == "":
					w2v_min_count = "1"
				if w2v_negative == "":
					w2v_negative = "5"
				if batch_size == "":
					batch_size = "128"
				if epochs == "":
					epochs = "40"
				if max_sequence_length == "":
					max_sequence_length = "5000"
				if num_filter == "":
					num_filter = "128"
				if drop_rate == "":
					drop_rate = "0.4"

				print("Start TEXTCNN Training")
				textcnn_result = TextcnnModel.textcnn_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, w2v_size, w2v_window, w2v_min_count, w2v_negative, batch_size, epochs, max_sequence_length, num_filter, drop_rate)
				print("End TEXTCNN Training")
				self.write(textcnn_result)
				# self.write('mlb_id:', textcnn_mlb_id)
				# self.write('tokenizer_id:', textcnn_tokenizer_id)
				# self.write('model_id:', textcnn_model_id)

			elif model_type == 'MLKNN':
				ngram_num = str(self.get_argument('ngram_num', ''))
				feature_num = str(self.get_argument('feature_num', ''))
				ml_k = str(self.get_argument('ml_k', ''))
				ml_s = str(self.get_argument('ml_s', ''))

				if ngram_num == "":
					ngram_num = "3"
				if feature_num == "":
					feature_num = "8000"
				if ml_k == "":
					ml_k = "50"
				if ml_s == "":
					ml_s = "1.0"

				print("Start MLKNN Training")
				ml_result = MLModel.knn_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, ml_k, ml_s)
				print("End MLKNN Training")
				self.write(ml_result)
				# self.write('tfidf_id:', knn_tfidf_id)
				# self.write('mlb_id:', knn_mlb_id)
				# self.write('model_id:', knn_classifier_id)

			elif model_type == 'AD_BR':
				ngram_num = str(self.get_argument('ngram_num', ''))
				feature_num = str(self.get_argument('feature_num', ''))
				samples_leaf = str(self.get_argument('samples_leaf', ''))
				samples_split = str(self.get_argument('samples_split', ''))

				if ngram_num == "":
					ngram_num = "3"
				if feature_num == "":
					feature_num = "8000"
				if samples_leaf == "":
					samples_leaf = "1"
				if samples_split == "":
					samples_split = "2"

				Logger.log_DEBUG.info("ngram_num: %s" % ngram_num)
				Logger.log_DEBUG.info("feature_num: %s" % feature_num)
				Logger.log_DEBUG.info("samples_leaf: %s" % samples_leaf)
				Logger.log_DEBUG.info("samples_split: %s" % samples_split)


				print("Start AD_BR Training")
				br_result = MLModel.br_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, samples_leaf, samples_split)
				print("End AD_BR Training")
				self.write(br_result)
				# self.write('tfidf_id:', br_tfidf_id)
				# self.write('mlb_id:', br_mlb_id)
				# self.write('model_id:', br_classifier_id)

			elif model_type == 'AD_CC':
				ngram_num = str(self.get_argument('ngram_num', ''))
				feature_num = str(self.get_argument('feature_num', ''))
				samples_leaf = str(self.get_argument('samples_leaf', ''))
				samples_split = str(self.get_argument('samples_split', ''))

				if ngram_num == "":
					ngram_num = "3"
				if feature_num == "":
					feature_num = "8000"
				if samples_leaf == "":
					samples_leaf = "1"
				if samples_split == "":
					samples_split = "2"

				print("Start AD_CC Training")
				cc_result = MLModel.cc_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, samples_leaf, samples_split)
				print("End AD_CC Training")
				self.write(cc_result)
				# self.write('tfidf_id:', cc_tfidf_id)
				# self.write('mlb_id:', cc_mlb_id)
				# self.write('model_id:', cc_classifier_id)

			elif model_type == 'AD_LP':
				ngram_num = str(self.get_argument('ngram_num', ''))
				feature_num = str(self.get_argument('feature_num', ''))
				samples_leaf = str(self.get_argument('samples_leaf', ''))
				samples_split = str(self.get_argument('samples_split', ''))

				if ngram_num == "":
					ngram_num = "3"
				if feature_num == "":
					feature_num = "8000"
				if samples_leaf == "":
					samples_leaf = "1"
				if samples_split == "":
					samples_split = "2"

				print("Start AD_LP Training")
				lp_result = MLModel.lp_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, samples_leaf, samples_split)
				print("End AD_LP Training")
				self.write(lp_result)
				# self.write('tfidf_id:', lp_tfidf_id)
				# self.write('mlb_id:', lp_mlb_id)
				# self.write('model_id:', lp_classifier_id)
				
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
	sockets = tornado.netutil.bind_sockets(8000)# 绑定端口创建套接字
	tornado.process.fork_processes(0)# 开启多进程，0表示自动识别CPU核心数
	server = tornado.httpserver.HTTPServer(app)# 创建http服务
	server.add_sockets(sockets)# 将套接字添加到http服务中
	tornado.ioloop.IOLoop.instance().start() # 启动io事件循环