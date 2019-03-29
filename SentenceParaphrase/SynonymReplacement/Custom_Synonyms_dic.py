# coding:utf-8

import os
import re
import synonyms
import time
import sys
import config_SR
from SynonymReplacement.my_log_new import Logger


class Synonyms_Dictionary:
    '''
    同义词获取类
    说明：初始化同义词林，自定义个体同义词，自带同义词包
    作者：仝哲
    时间：2018年5月4日
    '''

    def __init__(self, syDict={}, syDict_indiv={}, synonym_path='', synonym_indiv_path=''):
        '''
        同义词获取构造函数，当synonym_path为空字符串则不读取同义词林,当synonym_indiv_path为空字符串时不读取个体同义词数据
        :param syDict: 同义词典
        :param syDict_indiv: 同义词个体词典
        :param synonym_path: 同义词典读取路径
        :param synonym_indiv_path: 同义词个体词典路径
        '''
        # 哈工大词林词典（key:词名；value：同义词列表）
        self.synonyms_Dic = syDict
        # 个体同义词词典（key:词名；value：同义词列表）
        self.synonyms_indiv_Dic = syDict_indiv
        # 同义词词林读取路径
        self.synonym_path = synonym_path
        # 个体同义词读取路径
        self.synonym_indiv_path = synonym_indiv_path

        if synonym_path != "":
            self.Init_Synonyms_Dic(self.synonyms_Dic, self.synonym_path)
            Logger.log_DEBUG.info("同义词林初始化完成！词林字典大小%d" % len(self.synonyms_Dic))
        else:
            Logger.log_DEBUG.info("没有同义词林")

        if synonym_indiv_path != "":
            self.Init_Synonyms_indiv_Dic(self.synonyms_indiv_Dic, self.synonym_indiv_path)
            Logger.log_DEBUG.info("个体同义词初始化完成！同义词字典大小%d" % len(self.synonyms_indiv_Dic))
        else:
            Logger.log_DEBUG.info("没有个体同义词")

    def Init_Synonyms_Dic(self, syDict, synonym_path):
        '''
        初始化哈工大同义词词林到自定义字典结构中(使用Get_Synonyms_Word_Fores函数前需要先初始化)
        在构造函数中被调用
        :param syDict: 同义词典
        :param synonym_path: 同义词词典路径
        :return: 初始化后的同义词典
        '''
        try:
            if os.path.exists(synonym_path):
                fr = open(synonym_path, 'r', encoding='utf8')
                for line in fr.readlines():
                    if line:
                        line_temp = [w.replace('\n', '').replace('\r', '').replace('\t', '') for w in line.split(' ') if len(w) != 0]
                        if len(line_temp) > 1:
                            syDict[line_temp[0]] = line_temp[1:]
            else:
                Logger.log_DEBUG.info("找不到同义词文件,请检查传入路径是否正确")
        except MemoryError as e:
            s = "内存不够请检查同义词林大小，及电脑内存配置。" + str(e)
            Logger.log_ERROR.error(s)
            raise TypeError(s)
        except Exception as e:
            s = "初始化同义词典时发生异常Init_Synonyms_Dic" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)

    def Init_Synonyms_indiv_Dic(self, syDict, synonym_path):
        '''
        初始化个体同义词到自定义字典结构中
        在构造函数中被调用
        :param syDict: 同义词典
        :param synonym_path: 同义词词典路径
        :return: 初始化后的同义词典
        '''
        try:
            if os.path.exists(synonym_path):
                fr = open(synonym_path, 'r', encoding='utf8')
                for line in fr.readlines():
                    if line:
                        line_temp = [w.replace('\n', '').replace('\r', '').replace('\t', '') for w in line.split(' ') if
                                     len(w) != 0]
                        if len(line_temp) > 1:
                            for w in line_temp:
                                syDict[w] = line_temp
            else:
                Logger.log_DEBUG.info("找不到同义词文件,请检查传入路径是否正确")
        except MemoryError as e:
            s = "内存不够请检查个体同义词大小，及电脑内存配置。" + str(e)
            Logger.log_ERROR.e(s)
            raise TypeError(s)
        except Exception as e:
            s = "初始化同义词典时发生异常Init_Synonyms_indiv_Dic" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)

    def contain_zh(self, syslist):
        '''
        判断传入字符串是否包含中文
        :param syslist: 但判断字符串列表
        :return: 只包含中文的字符串列表
        '''
        zh_pattern = re.compile('[\u4e00-\u9fa5]+')
        result = [word for word in syslist if zh_pattern.search(word)]
        return result

    def Get_Synonyms_Word_Forest(self, Word):
        '''
        获取同义词列表(通过同义词词林)
        :param Word: 待取同义词词语
        :return: 返回Word同义词列表
        '''
        synonyms_list = []
        syDict = self.synonyms_Dic
        if Word in syDict:
            return self.contain_zh(syDict[Word])
        else:
            return synonyms_list

    def Get_Synonyms_indiv(self, Word):
        '''
        获取个体同义词列表
        :param Word: 待取同义词词语
        :return: 返回Word同义词列表
        '''
        synonyms_list = []
        syDict = self.synonyms_indiv_Dic
        if Word in syDict:
            return self.contain_zh(syDict[Word])
        else:
            return synonyms_list

    def Get_Synonyms_Package(self, Word):
        '''
        获取同义词列表(通过Python自带Synonyms包)
        :param Word: 待取同义词词语
        :return: 返回Word同义词列表
        '''
        try:
            # 选择前五个同义词，默认为十个
            temp_list, temp_value = synonyms.nearby(Word)
            if len(temp_list)!=0:
                synonyms_list = [temp_list[i] for i in range(10) if temp_value[i] >= sum(temp_value[1:9]) / 8]
            else:
                synonyms_list=[]
            return self.contain_zh(synonyms_list)
        except Exception as e:
            s = "调用Python自带同义词包时发生异常Get_Synonyms_Package" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)

    def Get_Synonyms(self, Word):
        '''
        获取同义词列表
        :param Word: 待取同义词词语
        :return: 返回Word同义词列表
        '''
        try:
            a = self.Get_Synonyms_Word_Forest(Word)
            b = self.Get_Synonyms_Package(Word)
            #a=[]
            return list(set(a).union(set(b)))
        except Exception as e:
            s = "同义词获取时发生异常Get_Synonyms" + str(e)
            #输出出错代码位置到日志
            Logger.log_ERROR.exception(sys.exc_info())
            Logger.log_ERROR.error(s)
            raise TypeError(s)


if __name__ == '__main__':
    start1 = time.time()
    # 新建实例
    SD = Synonyms_Dictionary({}, {}, config_SR.sys_path, config_SR.sys_indiv_path)
    print(config_SR.userdict_path)
    end1 = time.time()
    print(end1 - start1)
    print(SD.Get_Synonyms_Word_Forest('办理'))
    print(SD.Get_Synonyms_Package('办理'))
    print(SD.Get_Synonyms('办理'))
    print(SD.Get_Synonyms_indiv('办理'))
    # print(SD.synonyms_indiv_Dic)
