# coding:utf-8
import os
import jieba
import copy
import requests
import sys
import random
from itertools import combinations
from SynonymReplacement.Sentence_Segmentation import Senten_Seg
from SynonymReplacement.Custom_Synonyms_dic import Synonyms_Dictionary
from SynonymReplacement.my_log_new import Logger
import config_SR

class Synonym_Replace:
    '''
    同义词替换类
    说明：对待复述语句进行分词及词性标注，存入相应属性中方便同义词替换处理
    作者：钱艳
    时间：2018年5月3日
    '''

    def __init__(self):
        # 实例化同义词典
        self.sys_Dic = Synonyms_Dictionary({}, {}, config_SR.sys_path, config_SR.sys_indiv_path)

    def init_sys_dic(self):
        '''
        :return: 初始化结巴分词自定义词典
        '''
        # 加载自定义个体词典到到结巴分词中
        jieba.load_userdict(config_SR.userdict_path)

    def Synonym_Replace_mian(self, sentence, num=5):
        '''
        同义词替换主函数
        :param sentence: 待复述语句
        :param num: 通过语言模型返回句子个数，默认为5个
        :return: 返回复述结果列表
        '''
        try:
            paraphraser_Result = []
            # 对句子分词及词性标注
            senten_seg = Senten_Seg(sentence, self.sys_Dic)

            # 判断是否进行同义词替换
            if len(senten_seg.replace_Loc) > 4:
                Logger.log_DEBUG.info("可替换位置太多，请查看复述语句是否过长或分词是否合理，如果句子过长请修改为简单句，分词不理请维护自定义词典")
                paraphraser_Result.append(senten_seg.sentence)
                return paraphraser_Result

            for i in range(senten_seg.loc_num):
                # 从可替换位置中得到替换位置组合
                tem_list = list(combinations(senten_seg.replace_Loc, i + 1))
                for Loc in tem_list:
                    paraphraser_Result = paraphraser_Result + self.Start_Replace(senten_seg, Loc)
            Logger.log_DEBUG.info("同义词替换候选句子总数：%d" % len(paraphraser_Result))  # Logger.log.info(paraphraser_Result)
            #print(paraphraser_Result)
            new_Result = []
            if len(paraphraser_Result) > 0:
                # 调用语言模型得到排名靠前的n个句子
                new_Result.append(senten_seg.word)#加原句
                if len(paraphraser_Result)>config_SR.max_num:
                    # 从paraphraser_Result中随机获取config_SR.max_num个元素，作为一个片断返回
                    slice = random.sample(paraphraser_Result, config_SR.max_num)
                    new_Result = new_Result + self.Use_Language_Model(slice, num)
                else:
                    new_Result=new_Result+self.Use_Language_Model(paraphraser_Result, num)
            else:
                new_Result.append(senten_seg.word)

            # 判断主题是否需要替换，需要则替换不需要则直接返回结果
            result = []
            if len(senten_seg.indiv_name) > 0:
                # 主体同义词典
                sys_dic_indiv = senten_seg.indiv_sys[0]
                for senten in new_Result:
                    result.append(''.join(senten))
                    for s in sys_dic_indiv:
                        if senten_seg.indiv_name[0] != s:
                            senten[senten_seg.indiv_loc[0]] = s
                            result.append(''.join(senten))
            else:
                for senten in new_Result:
                    result.append(''.join(senten))
            return result

        except Exception as e:
            s = "同义词替换主函数运行发生异常Synonym_Replace_mian" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)

    def Use_Language_Model(self, paraphraser_Result, n):
        '''
        调用语言模型特出排名靠前的n个句子
        :param paraphraser_Result: 待排名句子列表
        :param n: 返回句子个数
        :return: 返回排名最高的前n个句子
        '''
        try:
            temp_list = []
            for s in paraphraser_Result:
                temp_list.append(' '.join(s))
            temp_str = '+'.join(temp_list)
            user_info = {'sentence': temp_str, 'num': str(n)}
            # user_info = {'sentence': '我清楚了','model_type':'2'}

            #r = requests.post("http://10.0.12.112:3000/", data=user_info)
            #r = requests.post("http://10.0.12.112:3000/", data=user_info)
            r = requests.post(config_SR.url, data=user_info)
            # print(r.text)

            temp_list2 = r.text.split("+")
            new_Result = []
            for s in temp_list2:
                new_Result.append(s.split(" "))
            Logger.log_DEBUG.info('语言模型判断结果')
            Logger.log_DEBUG.info(new_Result)
            return new_Result
        except Exception as e:
            s = "调用语言模型发生异常Use_Language_Model" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)

    def Start_Replace(self, senten_seg, Loc):
        '''
        开始同义词替换
        :param senten_seg: 句子分词对象
        :param Loc: 替换位置列表
        :return: 替换后的新句子列表
        '''
        try:
            new_result = []
            for i in Loc:
                # 得到可替换词的同义词
                sys = senten_seg.replace_sys[i]
                if i == Loc[0]:
                    for s in sys:
                        listb = copy.deepcopy(senten_seg.word)
                        if listb[i] != s:
                            listb[i] = s
                            new_result.append(listb)
                else:
                    temp = copy.deepcopy(new_result)
                    new_result = []
                    for n in temp:
                        for s in sys:
                            listb = copy.deepcopy(n)
                            if listb[i] != s:
                                listb[i] = s
                                new_result.append(listb)
            return new_result
        except Exception as e:
            s = "同义词替换时发生异常Start_Replace" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)


if __name__ == '__main__':
    sys_rep = Synonym_Replace()
    sys_rep.init_sys_dic()
    s = '贷记卡申请进度查询'
    #s = '你给我查下我信用卡办的怎么样了'
    result = sys_rep.Synonym_Replace_mian(s)
    print("生成复述语句：%d" % len(result))
    print(result)
