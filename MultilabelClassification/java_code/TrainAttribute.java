package com.utry.multilabel;

import java.util.HashMap;

/**
 * @Desctiption 分类训练类
 * @author molian
 * @time 2018/10/09
 */
public class TrainAttribute extends TrainIdentifyBasis {
	
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * TFIDF特征相关参数
	 */
	private HashMap<String, String> tfidfParam = new HashMap<String, String>();
	
	/**
	 * 词向量训练相关参数
	 */
	private HashMap<String, String> w2vParam = new HashMap<String, String>();
	
	/**
	 * 传统机器学习相关参数
	 */
	private HashMap<String, String> mlParam = new HashMap<String, String>();
	
	/**
	 * 深度学习模型相关参数
	 */
	private HashMap<String, String> nnParam = new HashMap<String, String>();
	
	/**
	 * 训练模型服务url
	 */
	private String train_service_url = new String("");
	
	
	/**
	 * 
	 * @return TFIDF特征相关参数 tfidfParam
	 */
	public HashMap<String, String> getTfidfParam() {
		return tfidfParam;
	}
	
	/**
	 * 
	 * @param tfidfParam TFIDF特征相关参数
	 */
	public void setTfidfParam(HashMap<String, String> tfidfParam) {
		this.tfidfParam = tfidfParam;
	}
	
	/**
	 * 
	 * @return 词向量训练相关参数 w2cParam
	 */
	public HashMap<String, String> getW2vParam() {
		return w2vParam;
	}
	
	/**
	 * 
	 * @param w2cParam 词向量训练相关参数
	 */
	public void setW2vParam(HashMap<String, String> w2vParam) {
		this.w2vParam = w2vParam;
	}
	
	/**
	 * 
	 * @return 传统机器学习相关参数 mlParam
	 */
	public HashMap<String, String> getMlParam() {
		return mlParam;
	}
	
	/**
	 * 
	 * @param mlParam 传统机器学习相关参数
	 */
	public void setMlParam(HashMap<String, String> mlParam) {
		this.mlParam = mlParam;
	}
	
	/**
	 * 
	 * @return 深度学习模型相关参数 nnParam
	 */
	public HashMap<String, String> getNnParam() {
		return nnParam;
	}
	
	/**
	 * 
	 * @param nnParam 深度学习模型相关参数
	 */
	public void setNnParam(HashMap<String, String> nnParam) {
		this.nnParam = nnParam;
	}
	
	/**
	 * 
	 * @return 训练模型服务url train_service_url
	 */
	public String getTrain_service_url() {
		return train_service_url;
	}
	
	/**
	 * 
	 * @param train_service_url 训练模型服务url
	 */
	public void setTrain_service_url(String train_service_url) {
		this.train_service_url = train_service_url;
	}
	
	public static void main(String[] args) {
		
	}

}
