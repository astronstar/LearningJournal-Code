package com.utry.multilabel;

import java.io.Serializable;
import java.util.HashMap;

/**
 * @Desctiption 训练和识别基类
 * @author molian
 * @time 2018/10/09
 */
public class TrainIdentifyBasis implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	/**
	 * 项目id
	 */
	private String itemid = "";
	
	/**
	 * 模型类型
	 */
	private String modelType = "";
	
	
	/**
	 * 文件服务相关参数
	 */
	private HashMap<String, String> fileParam = new HashMap<String, String>();
	
	/**
	 * 模型id
	 */
	private String modelid = "";
	
	/**
	 * 模型文件id
	 */
	private String modelfileid = "";
	
	/**
	 * mlb模型文件id
	 */
	private String mlbfileid = "";
	
	/**
	 * tokenizer模型文件id
	 */
	private String tokenizerfileid = "";
	
	/**
	 * tfidf模型文件id
	 */
	private String tfidffileid = "";
	
	/**
	 * 文件服务器操作相关参数
	 */
	private HashMap<String, String> fileserverParam = new HashMap<String, String>();
	
	/**
	 * @return 项目id
	 */
	public String getItemid() {
		return itemid;
	}

	/**
	 * @param itemid 项目id
	 */
	public void setItemid(String itemid) {
		this.itemid = itemid;
	}
	
	/**
	 * @return 模型类型
	 */
	public String getModelType() {
		return modelType;
	}
	
	/**
	 * @param modelType 模型类型
	 */
	public void setModelType(String modelType) {
		this.modelType = modelType;
	}
	
	/**
	 * 
	 * @return 文件服务相关参数 fileParam
	 */
	public HashMap<String, String> getFileParam() {
		return fileParam;
	}
	
	/**
	 * 
	 * @param fileParam 文件服务相关参数
	 */
	public void setFileParam(HashMap<String, String> fileParam) {
		this.fileParam = fileParam;
	}
	
	/**
	 * @return 模型id
	 */
	public String getModelid() {
		return modelid;
	}
	
	/**
	 * 
	 * @param modelfileid 模型文件id
	 */
	public void setModelfileid(String modelfileid) {
		this.modelfileid = modelfileid;
	}
	
	/**
	 * @return 模型文件id
	 */
	public String getModelfileid() {
		return modelfileid;
	}
	
	/**
	 * 
	 * @param modelid 模型id
	 */
	public void setModelid(String modelid) {
		this.modelid = modelid;
	}
	
	/**
	 * @return mlb模型id
	 */
	public String getMlbfileid() {
		return mlbfileid;
	}
	
	/**
	 * @param mlbid mlb模型id
	 */
	public void setMlbfileid(String mlbfileid) {
		this.mlbfileid = mlbfileid;
	}
	
	/**
	 * @return tokenizer模型id
	 */
	public String getTokenizerfileid() {
		return tokenizerfileid;
	}
	
	/**
	 * @param tokenizerid tokenizer模型id
	 */
	public void setTokenizerfileid(String tokenizerfileid) {
		this.tokenizerfileid = tokenizerfileid;
	}
	
	/**
	 * @return tfidf模型id
	 */
	public String getTfidffileid() {
		return tfidffileid;
	}
	
	/**
	 * @param tfidfid tfidf模型id
	 */
	public void setTfidffileid(String tfidffileid) {
		this.tfidffileid = tfidffileid;
	}
	
	public HashMap<String, String> getFileserverParam() {
		return fileserverParam;
	}

	public void setFileserverParam(HashMap<String, String> fileserverParam) {
		this.fileserverParam = fileserverParam;
	}
	
	public static void main(String[] args) {
		
	}
}
