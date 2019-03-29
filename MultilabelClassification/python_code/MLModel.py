# coding:utf-8

"""
	作者：	仝哲
	日期：	2018年10月22日
	文件名：	MLModel.py
	功能：	多标签分类——传统机器学习模型训练及保存
"""

import jieba
import pandas as pd
import pickle
from skmultilearn.adapt import MLkNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.metrics import jaccard_similarity_score, precision_score, recall_score, f1_score, accuracy_score, \
	hamming_loss
from DataLoad import Data_Load
from MyLog import Logger


def MLKNN_method(X_train, y_train, ml_k, ml_s):
	"""
	改编算法-->MLKNN方法
	:param X_train: 输入数据
	:param y_train: 对应标签数据
	:return:
	"""
	try:
		classifier = MLkNN(k=int(ml_k), s=float(ml_s))
		classifier.fit(X_train, y_train)

		return classifier
	except Exception as e:
		print("warning----改编算法KNN|MLKNN----" + str(e))

	return None

def BinaryRelevance_method(X_train, y_train, samples_leaf, samples_split):
	"""
	问题转换-->二元关联方法
	:param X_train: 输入数据
	:param y_train: 对应标签数据
	:return:
	"""
	try:
		classifier = BinaryRelevance(DecisionTreeClassifier(min_samples_leaf=int(samples_leaf), min_samples_split=int(samples_split)))
		classifier.fit(X_train, y_train)

		return classifier
	except Exception as e:
		print("warning----二元关联|BinaryRelevance_method----" + str(e))

	return None

def ClassifierChain_method(X_train, y_train, samples_leaf, samples_split):
	"""
	问题转换-->分类器链方法
	:param X_train: 输入数据
	:param y_train: 对应标签数据
	:return:
	"""
	try:
		classifier = ClassifierChain(DecisionTreeClassifier(min_samples_leaf=int(samples_leaf), min_samples_split=int(samples_split)))
		classifier.fit(X_train, y_train)

		return classifier
	except Exception as e:
		print("warning----分类器链|ClassifierChain_method----" + str(e))

	return None

def LabelPowerset_method(X_train, y_train, samples_leaf, samples_split):
	"""
	问题转换-->标签Powerset方法
	:param X_train: 输入数据
	:param y_train: 对应标签数据
	:return:
	"""
	try:
		classifier = LabelPowerset(DecisionTreeClassifier(min_samples_leaf=int(samples_leaf), min_samples_split=int(samples_split)))
		classifier.fit(X_train, y_train)
		return classifier
	except Exception as e:
		print("warning----标签Powerset|LabelPowerset_method----" + str(e))

	return None

def get_tfidf_model(data, ngram_num, feature_num):
	"""
	tfidf模型构建
	:param data: 输入数据
	:return:
	"""
	try:
		vec = TfidfVectorizer(ngram_range=(1, int(ngram_num)), max_df=0.95, min_df=3, max_features=int(feature_num))
		vec.fit(data)

		return vec
	except Exception as e:
		print("warning----tfidf特征生成----" + str(e))

	return None

def train_data_cut(data):
	"""
	数据预处理，标签切割及语句分词
	:param data: 原始数据DataFrame格式，列名为 id, context, label
	:return:
	"""
	data['label'] = data['label'].apply(lambda x: x.split(','))
	data['context'] = data['context'].apply(lambda x: ' '.join(jieba.cut(x)))

	return data

def predict_data_cut(data):
	"""
	数据预处理，标签切割及语句分词
	:param data: 原始数据DataFrame格式，列名为 id, context, label
	:return:
	"""
	# data['label'] = data['label'].apply(lambda x: x.split(','))
	data['context'] = data['context'].apply(lambda x: ' '.join(jieba.cut(x)))

	return data

def get_label_model(data):
	"""
	获得标签二元化处理模型
	:param data: 标签数据
	:return: 标签二元化处理后的模型
	"""
	try:
		mlb = MultiLabelBinarizer()
		mlb.fit(list(data))

		return mlb
	except Exception as e:
		print("warning----label转换----")

	return None

def get_metrics(y_test, predictions):
	"""
	评价指标结果输出
	:param y_test: 真实值
	:param predictions: 预测值
	:return:
	"""
	print('accuracy_score', accuracy_score(y_test, predictions))
	print('hamming_loss', hamming_loss(y_test, predictions))
	print('jaccard_score', jaccard_similarity_score(y_test, predictions))
	print('precision_score', precision_score(y_test, predictions, average='samples'))
	print('recall_score', recall_score(y_test, predictions, average='samples'))
	print('f1_score', f1_score(y_test, predictions, average='samples'))

def knn_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, ml_k, ml_s):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)

	train_data = dl.get_train_data(train_data_id)
	train_data = train_data_cut(train_data)

	knn_tfidf = get_tfidf_model(train_data['context'], ngram_num, feature_num)
	knn_tfidf_content = pickle.dumps(knn_tfidf)
	dl.get_api_ticket()
	knn_tfidf_id = dl.upload_file_by_data(knn_tfidf_content)
	print('knn_tfidf_id:', knn_tfidf_id)

	knn_mlb = get_label_model(train_data['label'])
	print('数据行数:', len(train_data['label']))

	knn_mlb_content = pickle.dumps(knn_mlb)
	dl.get_api_ticket()
	knn_mlb_id = dl.upload_file_by_data(knn_mlb_content)
	print('knn_mlb_id:', knn_mlb_id)

	feat = knn_tfidf.transform(train_data['context'])
	label = knn_mlb.transform(train_data['label'])

	knn_classifier = MLKNN_method(feat, label, ml_k, ml_s)
	knn_classifier_content = pickle.dumps(knn_classifier)
	dl.get_api_ticket()
	knn_model_id = dl.upload_file_by_data(knn_classifier_content)
	print('knn_model_id:', knn_model_id)

	result = "tfidf_id#mlb_id#classifier_id##" + str(knn_tfidf_id) + "#" + str(knn_mlb_id) + "#" + str(knn_model_id)

	return result

def knn_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, knn_tfidf_id, knn_mlb_id, knn_model_id):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	data = dl.get_train_data(data_id)
	data = predict_data_cut(data)

	dl.get_api_ticket()
	knn_tfidf = pickle.loads(dl.download_model_by_id(knn_tfidf_id))

	dl.get_api_ticket()
	knn_mlb = pickle.loads(dl.download_model_by_id(knn_mlb_id))

	dl.get_api_ticket()
	knn_model = pickle.loads(dl.download_model_by_id(knn_model_id))

	data_vec = knn_tfidf.transform(data['context'])

	y_pred = knn_model.predict(data_vec)
	y_pred = (y_pred > 0.5)
	# print(y_pred)

	label = [','.join(i) for i in knn_mlb.inverse_transform(y_pred)]
	# print(label)

	ans = {'id': data['id'], 'label': label}
	ans = pd.DataFrame(ans)
	# print(ans)

	ans['result'] = ans['id'].map(str) + "#" + ans['label'].map(str)
	result_list = ans['result'].tolist()
	
	result = ''
	for i in result_list:
		result = i + "##" + result

	# print('返回结果:', result[:-2])

	return result[:-2]

def br_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, samples_leaf, samples_split):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	train_data = dl.get_train_data(train_data_id)
	train_data = train_data_cut(train_data)

	br_tfidf = get_tfidf_model(train_data['context'], ngram_num, feature_num)

	br_tfidf_content = pickle.dumps(br_tfidf)
	dl.get_api_ticket()
	br_tfidf_id = dl.upload_file_by_data(br_tfidf_content)
	print('br_tfidf_id:', br_tfidf_id)

	br_mlb = get_label_model(train_data['label'])
	print('数据行数:', len(train_data['label']))

	br_mlb_content = pickle.dumps(br_mlb)
	dl.get_api_ticket()
	br_mlb_id = dl.upload_file_by_data(br_mlb_content)
	print('br_mlb_id:', br_mlb_id)

	feat = br_tfidf.transform(train_data['context'])
	label = br_mlb.transform(train_data['label'])

	br_classifier = BinaryRelevance_method(feat, label, samples_leaf, samples_split)
	br_classifier_content = pickle.dumps(br_classifier)
	dl.get_api_ticket()
	br_model_id = dl.upload_file_by_data(br_classifier_content)
	print('br_model_id:', br_model_id)

	result = "tfidf_id#mlb_id#model_id##" + str(br_tfidf_id) + "#" + str(br_mlb_id) + "#" + str(br_model_id)

	return result

def br_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, br_tfidf_id, br_mlb_id, br_model_id):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	data = dl.get_train_data(data_id)
	data = predict_data_cut(data)

	dl.get_api_ticket()
	br_tfidf = pickle.loads(dl.download_model_by_id(br_tfidf_id))

	dl.get_api_ticket()
	br_mlb = pickle.loads(dl.download_model_by_id(br_mlb_id))

	dl.get_api_ticket()
	br_model = pickle.loads(dl.download_model_by_id(br_model_id))

	data_vec = br_tfidf.transform(data['context'])

	y_pred = br_model.predict(data_vec)
	y_pred = (y_pred > 0.5)
	# print(y_pred)

	label = [','.join(i) for i in br_mlb.inverse_transform(y_pred)]
	# print("标签结果：",label)

	ans = {'id': data['id'], 'label': label}
	ans = pd.DataFrame(ans)
	# print(ans)

	ans['result'] = ans['id'].map(str) + "#" + ans['label'].map(str)
	result_list = ans['result'].tolist()
	
	result = ''
	for i in result_list:
		result = i + "##" + result

	# print("返回结果：",result[:-2])

	return result[:-2]

def cc_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, samples_leaf, samples_split):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	train_data = dl.get_train_data(train_data_id)
	train_data = train_data_cut(train_data)

	cc_tfidf = get_tfidf_model(train_data['context'], ngram_num, feature_num)
	cc_tfidf_content = pickle.dumps(cc_tfidf)
	dl.get_api_ticket()
	cc_tfidf_id = dl.upload_file_by_data(cc_tfidf_content)
	print('cc_tfidf_id:', cc_tfidf_id)

	cc_mlb = get_label_model(train_data['label'])
	print('数据行数:', len(train_data['label']))

	cc_mlb_content = pickle.dumps(cc_mlb)
	dl.get_api_ticket()
	cc_mlb_id = dl.upload_file_by_data(cc_mlb_content)
	print('cc_mlb_id:', cc_mlb_id)

	feat = cc_tfidf.transform(train_data['context'])
	label = cc_mlb.transform(train_data['label'])

	cc_classifier = ClassifierChain_method(feat, label, samples_leaf, samples_split)
	cc_classifier_content = pickle.dumps(cc_classifier)
	dl.get_api_ticket()
	cc_model_id = dl.upload_file_by_data(cc_classifier_content)
	print('cc_model_id:', cc_model_id)

	result = "tfidf_id#mlb_id#model_id##" + str(cc_tfidf_id) + "#" + str(cc_mlb_id) + "#" + str(cc_model_id)

	return result

def cc_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, cc_tfidf_id, cc_mlb_id, cc_model_id):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	data = dl.get_train_data(data_id)
	data = predict_data_cut(data)

	dl.get_api_ticket()
	cc_tfidf = pickle.loads(dl.download_model_by_id(cc_tfidf_id))

	dl.get_api_ticket()
	cc_mlb = pickle.loads(dl.download_model_by_id(cc_mlb_id))

	dl.get_api_ticket()
	cc_model = pickle.loads(dl.download_model_by_id(cc_model_id))

	data_vec = cc_tfidf.transform(data['context'])

	y_pred = cc_model.predict(data_vec)
	y_pred = (y_pred > 0.5)
	# print(y_pred)

	label = [','.join(i) for i in cc_mlb.inverse_transform(y_pred)]
	# print(label)

	ans = {'id': data['id'], 'label': label}
	ans = pd.DataFrame(ans)
	# print(ans)

	ans['result'] = ans['id'].map(str) + "#" + ans['label'].map(str)
	result_list = ans['result'].tolist()
	
	result = ''
	for i in result_list:
		result = i + "##" + result

	return result[:-2]

def lp_train(ip, up_url, down_url, access_url, access_key, _init_companyId, train_data_id, ngram_num, feature_num, samples_leaf, samples_split):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	train_data = dl.get_train_data(train_data_id)
	train_data = train_data_cut(train_data)

	lp_tfidf = get_tfidf_model(train_data['context'], ngram_num, feature_num)
	lp_tfidf_content = pickle.dumps(lp_tfidf)
	dl.get_api_ticket()
	lp_tfidf_id = dl.upload_file_by_data(lp_tfidf_content)
	print('lp_tfidf_id:', lp_tfidf_id)

	lp_mlb = get_label_model(train_data['label'])
	print('数据行数:', len(train_data['label']))

	lp_mlb_content = pickle.dumps(lp_mlb)
	dl.get_api_ticket()
	lp_mlb_id = dl.upload_file_by_data(lp_mlb_content)
	print('lp_mlb_id:', lp_mlb_id)

	feat = lp_tfidf.transform(train_data['context'])
	label = lp_mlb.transform(train_data['label'])

	lp_classifier = LabelPowerset_method(feat, label, int(samples_leaf), int(samples_split))
	lp_classifier_content = pickle.dumps(lp_classifier)
	dl.get_api_ticket()
	lp_model_id = dl.upload_file_by_data(lp_classifier_content)
	print('lp_model_id:', lp_model_id)

	result = "tfidf_id#mlb_id#model_id##" + str(lp_tfidf_id) + "#" + str(lp_mlb_id) + "#" + str(lp_model_id)

	return result

def lp_predict(ip, up_url, down_url, access_url, access_key, _init_companyId, data_id, lp_tfidf_id, lp_mlb_id, lp_model_id):
	dl = Data_Load(ip, up_url, down_url, access_url, access_key, _init_companyId)
	data = dl.get_train_data(data_id)
	data = predict_data_cut(data)

	dl.get_api_ticket()
	lp_tfidf = pickle.loads(dl.download_model_by_id(lp_tfidf_id))

	dl.get_api_ticket()
	lp_mlb = pickle.loads(dl.download_model_by_id(lp_mlb_id))

	dl.get_api_ticket()
	lp_model = pickle.loads(dl.download_model_by_id(lp_model_id))

	data_vec = lp_tfidf.transform(data['context'])

	y_pred = lp_model.predict(data_vec)
	y_pred = (y_pred > 0.5)
	# print(y_pred)

	label = [','.join(i) for i in lp_mlb.inverse_transform(y_pred)]
	# print(label)

	ans = {'id': data['id'], 'label': label}
	ans = pd.DataFrame(ans)
	# print(ans)

	ans['result'] = ans['id'].map(str) + "#" + ans['label'].map(str)
	result_list = ans['result'].tolist()
	
	result = ''
	for i in result_list:
		result = i + "##" + result

	return result[:-2]

if __name__ == '__main__':
	train_data_id = "7237d9e6d2b011e8be95000c2961e520"
	ngram_num = 3
	feature_num = 8000
	samples_leaf = 1
	samples_split = 2

	br_train(train_data_id, ngram_num, feature_num, samples_leaf, samples_split)