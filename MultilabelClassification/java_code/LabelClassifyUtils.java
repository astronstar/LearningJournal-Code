package com.utry.multilabel;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;


/**
 * @Desctiption 多标签分类工具类
 * @author molian
 * @time 2018/10/15
 */
public class LabelClassifyUtils implements Serializable {

	private static final long serialVersionUID = 1L;
	
	/**
	 * 模型训练接口
	 * 
	 * @param trainAttribute 分类训练类
	 * @throws Exception
	 */
	public String startTraining(TrainAttribute trainAttribute) throws Exception {
		
		String item_id = trainAttribute.getItemid();
		String train_data_id = trainAttribute.getFileParam().get("FILE_ID");
		String model_type = trainAttribute.getModelType();
		String train_service_url = trainAttribute.getTrain_service_url();
		
		System.out.println(train_service_url);
		
		String ip = trainAttribute.getFileserverParam().get("IP");
		String up_url = trainAttribute.getFileserverParam().get("UP_URL");
		String down_url = trainAttribute.getFileserverParam().get("DOWN_URL");
		String access_url = trainAttribute.getFileserverParam().get("ACCESS_URL");
		String access_key = trainAttribute.getFileserverParam().get("ACCESS_KEY");
		String _init_companyId = trainAttribute.getFileserverParam().get("INIT_COMPANYID");
		
		String content = "";
		if ("GRU".equals(model_type) || "TEXTCNN".equals(model_type)) {
			String w2v_size = trainAttribute.getW2vParam().get("W2V_SIZE");
			String w2v_window = trainAttribute.getW2vParam().get("W2V_WINDOW");
			String w2v_min_count = trainAttribute.getW2vParam().get("W2V_MIN_COUNT");
			String w2v_negative = trainAttribute.getW2vParam().get("W2V_NEGATIVE");
			String batch_size = trainAttribute.getNnParam().get("NN_BATCHSIZE");
			String epochs = trainAttribute.getNnParam().get("NN_EPOCHS");
			String max_sequence_length = trainAttribute.getNnParam().get("NN_MAX_SEQUENCE_LENGTH");
			String num_filter = trainAttribute.getNnParam().get("NN_NUM_FILTER");
			String drop_rate = trainAttribute.getNnParam().get("NN_DROP_RATE");
			
			content += "item_id=" + URLEncoder.encode(item_id, "UTF-8");
			
			content += "&ip=" + URLEncoder.encode(ip, "UTF-8");
			content += "&up_url=" + URLEncoder.encode(up_url, "UTF-8");
			content += "&down_url=" + URLEncoder.encode(down_url, "UTF-8");
			content += "&access_url=" + URLEncoder.encode(access_url, "UTF-8");
			content += "&access_key=" + URLEncoder.encode(access_key, "UTF-8");
			content += "&_init_companyId=" + URLEncoder.encode(_init_companyId, "UTF-8");
			
			content += "&model_type=" + URLEncoder.encode(model_type, "UTF-8");
			content += "&train_data_id=" + URLEncoder.encode(train_data_id, "UTF-8");
			content += "&w2v_size=" + URLEncoder.encode(w2v_size, "UTF-8");
			content += "&w2v_window=" + URLEncoder.encode(w2v_window, "UTF-8");
			content += "&w2v_min_count=" + URLEncoder.encode(w2v_min_count, "UTF-8");
			content += "&w2v_negative=" + URLEncoder.encode(w2v_negative, "UTF-8");
			content += "&batch_size=" + URLEncoder.encode(batch_size, "UTF-8");
			content += "&epochs=" + URLEncoder.encode(epochs, "UTF-8");
			content += "&max_sequence_length=" + URLEncoder.encode(max_sequence_length, "UTF-8");
			content += "&num_filter=" + URLEncoder.encode(num_filter, "UTF-8");
			content += "&drop_rate=" + URLEncoder.encode(drop_rate, "UTF-8");
		} else if ("MLKNN".equals(model_type)) {
			String ngram_num = trainAttribute.getTfidfParam().get("TFIDF_NGRAM_RANGE_NUM");
			String feature_num = trainAttribute.getTfidfParam().get("TFIDF_MAX_FEATURES_NUM");
			String ml_k = trainAttribute.getMlParam().get("ML_K");
			String ml_s = trainAttribute.getMlParam().get("ML_S");
			
			content += "item_id=" + URLEncoder.encode(item_id, "UTF-8");
			
			content += "&ip=" + URLEncoder.encode(ip, "UTF-8");
			content += "&up_url=" + URLEncoder.encode(up_url, "UTF-8");
			content += "&down_url=" + URLEncoder.encode(down_url, "UTF-8");
			content += "&access_url=" + URLEncoder.encode(access_url, "UTF-8");
			content += "&access_key=" + URLEncoder.encode(access_key, "UTF-8");
			content += "&_init_companyId=" + URLEncoder.encode(_init_companyId, "UTF-8");
			
			content += "&model_type=" + URLEncoder.encode(model_type, "UTF-8");
			content += "&train_data_id=" + URLEncoder.encode(train_data_id, "UTF-8");
			content += "&ngram_num=" + URLEncoder.encode(ngram_num, "UTF-8");
			content += "&feature_num=" + URLEncoder.encode(feature_num, "UTF-8");
			content += "&ml_k=" + URLEncoder.encode(ml_k, "UTF-8");
			content += "&ml_s=" + URLEncoder.encode(ml_s, "UTF-8");
		} else if ("AD_BR".equals(model_type) || "AD_CC".equals(model_type)  || "AD_LP".equals(model_type)) {
			String ngram_num = trainAttribute.getTfidfParam().get("TFIDF_NGRAM_RANGE_NUM");
			String feature_num = trainAttribute.getTfidfParam().get("TFIDF_MAX_FEATURES_NUM");
			String samples_leaf = trainAttribute.getMlParam().get("ML_MIN_SAMPLES_LEAF");
			String samples_split = trainAttribute.getMlParam().get("ML_MIN_SAMPLES_SPLIT");
			
			content += "item_id=" + URLEncoder.encode(item_id, "UTF-8");
			
			content += "&ip=" + URLEncoder.encode(ip, "UTF-8");
			content += "&up_url=" + URLEncoder.encode(up_url, "UTF-8");
			content += "&down_url=" + URLEncoder.encode(down_url, "UTF-8");
			content += "&access_url=" + URLEncoder.encode(access_url, "UTF-8");
			content += "&access_key=" + URLEncoder.encode(access_key, "UTF-8");
			content += "&_init_companyId=" + URLEncoder.encode(_init_companyId, "UTF-8");
			
			content += "&model_type=" + URLEncoder.encode(model_type, "UTF-8");
			content += "&train_data_id=" + URLEncoder.encode(train_data_id, "UTF-8");
			content += "&ngram_num=" + URLEncoder.encode(ngram_num, "UTF-8");
			content += "&feature_num=" + URLEncoder.encode(feature_num, "UTF-8");
			content += "&samples_leaf=" + URLEncoder.encode(samples_leaf, "UTF-8");
			content += "&samples_split=" + URLEncoder.encode(samples_split, "UTF-8");
		}

		HttpURLConnection connection = null;
		DataOutputStream out = null;
		BufferedReader reader = null;
		
		try {
			URL postUrl = new URL(train_service_url);
			connection = (HttpURLConnection) postUrl.openConnection();
			
			connection.setConnectTimeout(30000);
			connection.setReadTimeout(21600000);
			
			connection.setDoOutput(true);
			connection.setDoInput(true);
			
			connection.setRequestMethod("POST");
			
			connection.setUseCaches(false);
			
			connection.setInstanceFollowRedirects(true);
			
			connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
			
			connection.connect();
			
			out = new DataOutputStream(connection.getOutputStream());
			
			out.writeBytes(content);
			out.flush();
			out.close();
			
			reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
			String line;
			while ((line = reader.readLine()) != null) {
				reader.close();
				connection.disconnect();
				System.out.println(line);
				return line;
			}
			reader.close();
			connection.disconnect();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (out != null) {
				out.close();
			}
			if (reader != null) {
				reader.close();
			}
			if (connection != null) {
				connection.disconnect();
			}
		}
		return "fail";
	}
	
	/**
	 * 模型识别接口
	 * 
	 * @param recognizing 分类识别类
	 * @return 返回识别结果的标问ID列表及标问id对应的标问文本内容
	 * @throws Exception
	 */
	public HashMap<String, String> startIdentify(Identify identify) throws Exception {
		
		String item_id = identify.getItemid();
		String model_id = identify.getModelid();
		String model_type = identify.getModelType();
		String data_id = identify.getFileParam().get("FILE_ID");
		String mlb_file_id = identify.getMlbfileid();
		String model_file_id = identify.getModelfileid();
		String recognize_service_url = identify.getIdentify_service_url();
		
		System.out.println(recognize_service_url);
		
		String ip = identify.getFileserverParam().get("IP");
		String up_url = identify.getFileserverParam().get("UP_URL");
		String down_url = identify.getFileserverParam().get("DOWN_URL");
		String access_url = identify.getFileserverParam().get("ACCESS_URL");
		String access_key = identify.getFileserverParam().get("ACCESS_KEY");
		String _init_companyId = identify.getFileserverParam().get("INIT_COMPANYID");
		
		HashMap<String, String> result = new HashMap<String, String>();
		String content = "";	
		if ("GRU".equals(model_type)|| "TEXTCNN".equals(model_type)) {
			String tokenizer_file_id = identify.getTokenizerfileid();
			String max_sequence_length = identify.getMax_sequence_length();
			String batch_size = identify.getBatch_size();
			
			content += "item_id=" + URLEncoder.encode(item_id, "UTF-8");
			
			content += "&ip=" + URLEncoder.encode(ip, "UTF-8");
			content += "&up_url=" + URLEncoder.encode(up_url, "UTF-8");
			content += "&down_url=" + URLEncoder.encode(down_url, "UTF-8");
			content += "&access_url=" + URLEncoder.encode(access_url, "UTF-8");
			content += "&access_key=" + URLEncoder.encode(access_key, "UTF-8");
			content += "&_init_companyId=" + URLEncoder.encode(_init_companyId, "UTF-8");
			
			content += "model_id=" + URLEncoder.encode(model_id, "UTF-8");
			content += "&model_type=" + URLEncoder.encode(model_type, "UTF-8");
			content += "&data_id=" + URLEncoder.encode(data_id, "UTF-8");
			content += "&mlb_file_id=" + URLEncoder.encode(mlb_file_id, "UTF-8");
			content += "&tokenizer_file_id=" + URLEncoder.encode(tokenizer_file_id, "UTF-8");
			content += "&model_file_id=" + URLEncoder.encode(model_file_id, "UTF-8");
			content += "&max_sequence_length=" + URLEncoder.encode(max_sequence_length, "UTF-8");
			content += "&batch_size=" + URLEncoder.encode(batch_size, "UTF-8");
		} else if ("MLKNN".equals(model_type)  || "AD_BR".equals(model_type) || "AD_CC".equals(model_type) || "AD_LP".equals(model_type) ) {
			String tfidf_file_id = identify.getTfidffileid();
			
			content += "item_id=" + URLEncoder.encode(item_id, "UTF-8");
			
			content += "&ip=" + URLEncoder.encode(ip, "UTF-8");
			content += "&up_url=" + URLEncoder.encode(up_url, "UTF-8");
			content += "&down_url=" + URLEncoder.encode(down_url, "UTF-8");
			content += "&access_url=" + URLEncoder.encode(access_url, "UTF-8");
			content += "&access_key=" + URLEncoder.encode(access_key, "UTF-8");
			content += "&_init_companyId=" + URLEncoder.encode(_init_companyId, "UTF-8");
			
			content += "&model_type=" + URLEncoder.encode(model_type, "UTF-8");
			content += "&data_id=" + URLEncoder.encode(data_id, "UTF-8");
			content += "&tfidf_file_id=" + URLEncoder.encode(tfidf_file_id, "UTF-8");
			content += "&mlb_file_id=" + URLEncoder.encode(mlb_file_id, "UTF-8");
			content += "&model_file_id=" + URLEncoder.encode(model_file_id, "UTF-8");
		}
		
		String line = "";
		HttpURLConnection connection = null;
		DataOutputStream out = null;
		BufferedReader reader = null;
		
		try {
			// Post请求的post_url，与get不同的是不需要带参数
			URL postUrl = new URL(recognize_service_url);
			// 打开连接
			connection = (HttpURLConnection) postUrl.openConnection();

			connection.setConnectTimeout(30000);
			connection.setReadTimeout(21600000);

			// 设置是否向connection输出，因为这个是post请求，参数要放在
			// http正文内，因此需要设为true
			connection.setDoOutput(true);
			// Read from the connection. Default is true.
			connection.setDoInput(true);
			// 默认是 GET方式
			connection.setRequestMethod("POST");

			// Post 请求不能使用缓存
			connection.setUseCaches(false);

			connection.setInstanceFollowRedirects(true);

			// 配置本次连接的Content-type，配置为application/x-www-form-urlencoded的
			// 意思是正文是urlencoded编码过的form参数，下面我们可以看到我们对正文内容使用URLEncoder.encode
			// 进行编码
			connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
			// 连接，从postUrl.openConnection()至此的配置必须要在connect之前完成，
			// 要注意的是connection.getOutputStream会隐含的进行connect。
			connection.connect();
			out = new DataOutputStream(connection.getOutputStream());
			// The URL-encoded contend

			// DataOutputStream.writeBytes将字符串中的16位的unicode字符以8位的字符形式写到流里面
			out.writeBytes(content);
			out.flush();
			out.close();

			reader = new BufferedReader(new InputStreamReader(connection.getInputStream(), "utf-8"));
			while ((line = reader.readLine()) != null) {
				reader.close();
				connection.disconnect();

				String[] ans = line.split("##");
				for (int i = 0; i < ans.length; i++) {
					if (ans[i] != null) {
						String[] temp = ans[i].split("#", -1);
						result.put(temp[0], temp[1]);
					}
				}
				System.out.println(result);
				return result;
			}

			reader.close();
			connection.disconnect();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (out != null) {
				out.close();
			}
			if (reader != null) {
				reader.close();
			}
			if (connection != null) {
				connection.disconnect();
			}
		}
		return result;
	}
	
	public static void test_training() throws Exception {
		TrainAttribute trainAttribute = new TrainAttribute();
		
		trainAttribute.setItemid("1");
		trainAttribute.setModelid("100");
		trainAttribute.setModelType("AD_BR");
		
		HashMap<String, String> fileParam = new HashMap<String, String>();
//		fileParam.put("FILE_ID", "87ea99c4d11411e8be95000c2961e520");
		fileParam.put("FILE_ID", "7237d9e6d2b011e8be95000c2961e520");
		
		trainAttribute.setFileParam(fileParam);
		
		HashMap<String, String> fileserverParam = new HashMap<String, String>();
		fileserverParam.put("IP", "10.0.2.121:80");
		fileserverParam.put("UP_URL", "/fileManager/api/file/uploadFileP");
		fileserverParam.put("DOWN_URL", "/fileManager/api/file/downloadFileP");
		fileserverParam.put("ACCESS_URL", "/api/gateway/ticket");
		fileserverParam.put("ACCESS_KEY", "33a832d7949c11e89024000c2961e520");
		fileserverParam.put("INIT_COMPANYID", "08d181119a7b4c0e94ff368942fd4420");
		
		trainAttribute.setFileserverParam(fileserverParam);
		
		HashMap<String, String> w2vParam = new HashMap<String, String>();
		w2vParam.put("W2V_SIZE", "300");
		w2vParam.put("W2V_WINDOW", "5");
		w2vParam.put("W2V_MIN_COUNT", "1");
		w2vParam.put("W2V_NEGATIVE", "5");
		
		trainAttribute.setW2vParam(w2vParam);
		
		HashMap<String, String> nnParam = new HashMap<String, String>();
		nnParam.put("NN_BATCHSIZE", "128");
		nnParam.put("NN_EPOCHS", "1"); // 训练迭代次数
		nnParam.put("NN_MAX_SEQUENCE_LENGTH", "5000");
		nnParam.put("NN_NUM_FILTER", "128");
		nnParam.put("NN_DROP_RATE", "0.4");
		
		trainAttribute.setNnParam(nnParam);
		
		HashMap<String, String> mlParam = new HashMap<String, String>();
		mlParam.put("ML_K", "50");
		mlParam.put("ML_S", "1.0");
		mlParam.put("ML_MIN_SAMPLES_LEAF", "1");
		mlParam.put("ML_MIN_SAMPLES_SPLIT", "2");
		
		trainAttribute.setMlParam(mlParam);
		
		HashMap<String, String> tfidfParam = new HashMap<String, String>();
		tfidfParam.put("TFIDF_NGRAM_RANGE_NUM", "3");
		tfidfParam.put("TFIDF_MAX_FEATURES_NUM", "8000");
		
		trainAttribute.setTfidfParam(tfidfParam);
		
//		trainAttribute.setTrain_service_url("http://192.168.237.132:8000/");
		trainAttribute.setTrain_service_url("http://10.0.12.113:8000/");
		
		LabelClassifyUtils LCU = new LabelClassifyUtils();
		LCU.startTraining(trainAttribute);
	}
	
	public static void test_identify() throws Exception {
		Identify identify = new Identify();
		
		identify.setItemid("");
		identify.setModelid("");
		identify.setModelType("AD_BR");
		
		identify.setTfidffileid("3749511ad13a11e8be95000c2961e520");
		
		identify.setMlbfileid("38194b4cd13a11e8be95000c2961e520");
		identify.setModelfileid("2ecd9daed13c11e8be95000c2961e520");
		
		
//		identify.setTokenizerfileid("");
//		identify.setMax_sequence_length("5000");
//		identify.setBatch_size("128");

		HashMap<String, String> fileParam = new HashMap<String, String>();
		fileParam.put("FILE_ID", "87ea99c4d11411e8be95000c2961e520");
		
		identify.setFileParam(fileParam);
		
		HashMap<String, String> fileserverParam = new HashMap<String, String>();
		fileserverParam.put("IP", "10.0.2.121:80");
		fileserverParam.put("UP_URL", "/fileManager/api/file/uploadFileP");
		fileserverParam.put("DOWN_URL", "/fileManager/api/file/downloadFileP");
		fileserverParam.put("ACCESS_URL", "/api/gateway/ticket");
		fileserverParam.put("ACCESS_KEY", "33a832d7949c11e89024000c2961e520");
		fileserverParam.put("INIT_COMPANYID", "08d181119a7b4c0e94ff368942fd4420");
		
		identify.setFileserverParam(fileserverParam);
		
		identify.setIdentify_service_url("http://10.0.12.113:9000/");
		
		LabelClassifyUtils LCU = new LabelClassifyUtils();
		LCU.startIdentify(identify);
	}
	
	
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		// 文本多标签分类训练接口测试
		try {
			test_training();
		} catch (Exception e1) {
			e1.printStackTrace();
			System.out.println(e1.getMessage());
		}
		
		// 文本多标签分类识别接口测试
//		try {
//			test_identify();
//		} catch (Exception e2) {
//			e2.printStackTrace();
//			System.out.println(e2.getMessage());
//		}
		
		long endTime = System.currentTimeMillis(); // 获取结束时间
		System.out.println("第" + '0' + "次程序运行时间：" + (endTime - startTime));
	}
	
	
}
