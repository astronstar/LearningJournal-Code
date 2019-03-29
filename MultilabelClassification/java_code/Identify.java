package com.utry.multilabel;

import java.util.HashMap;

/**
 * @Desctiption 分类识别类
 * @author molian
 * @time 2018/10/09
 */
public class Identify extends TrainIdentifyBasis {

	private static final long serialVersionUID = 1L;


	/**
	 * 识别模型服务url
	 */
	private String identify_service_url = new String("");
	
	/**
	 * 待识别文本序列
	 */
	private HashMap<String, String> text = new HashMap<String, String>();
	
	/**
	 * max_sequence_length
	 */
	private String max_sequence_length = new String("");
	
	/**
	 * batch_size
	 */
	private String batch_size = new String("");

	/**
	 * @return 识别模型服务url recognize_service_url
	 */
	public String getIdentify_service_url() {
		return identify_service_url;
	}

	/**
	 * @param recognize_service_url 识别模型服务url
	 */
	public void setIdentify_service_url(String identify_service_url) {
		this.identify_service_url = identify_service_url;
	}
	
	
	public HashMap<String, String> getText() {
		return text;
	}

	public void setText(HashMap<String, String> text) {
		this.text = text;
	}

	public String getMax_sequence_length() {
		return max_sequence_length;
	}

	public void setMax_sequence_length(String max_sequence_length) {
		this.max_sequence_length = max_sequence_length;
	}

	public String getBatch_size() {
		return batch_size;
	}

	public void setBatch_size(String batch_size) {
		this.batch_size = batch_size;
	}

	public static void main(String[] args) {

	}
}
