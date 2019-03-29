# coding:utf-8
import jieba
import jieba.posseg
import sys
import config_SR
from SynonymReplacement.Custom_Synonyms_dic import Synonyms_Dictionary
from SynonymReplacement.my_log_new import Logger


class Senten_Seg:
    '''
    句子分词类
    说明：对待复述语句进行分词及词性标注，存入相应属性中方便同义词替换处理
    作者：钱艳
    时间：2018年5月2日
    '''

    # 可替换词性
    replace_Vocabulary = ['n', 'v', "vn"]

    # replace_Vocabulary = ['n', 'v']

    def __init__(self, sentence, sys_dic):
        '''
        句子分词类构造函数
        :param sentence: 待分词句子
        :param sys_dic: 同义词获取类实例
        '''
        try:
            # 原句
            self.sentence = sentence
            # 分词结果
            self.word = []
            # 分词词性
            self.vocabulary = []
            # 可替换词位置
            self.replace_Loc = []
            # 可替换词对应的同义词
            self.replace_sys = []
            # 主体名
            self.indiv_name = []
            # 主体位置
            self.indiv_loc = []
            # 主体同义词
            self.indiv_sys = []
            # 分词和词性标注
            # import jieba.posseg
            result = jieba.posseg.cut(sentence)
            loc = 0;
            for w in result:
                self.word.append(w.word)
                self.vocabulary.append(w.flag)
                if w.flag in Senten_Seg.replace_Vocabulary:
                    self.replace_Loc.append(loc)
                    self.replace_sys.append(sys_dic.Get_Synonyms(w.word))
                else:
                    self.replace_sys.append([])
                # print(w.word, "/", w.flag, ", ")
                if w.flag == "indiv":
                    self.indiv_name.append(w.word)
                    self.indiv_loc.append(loc)
                    self.indiv_sys.append(sys_dic.Get_Synonyms_indiv(w.word))
                loc = loc + 1
            # 可替换词个数
            self.loc_num = len(self.replace_Loc)
            Logger.log_DEBUG.info(self.sentence + " 完成句子分词")
            Logger.log_DEBUG.info("分词结果为：")
            Logger.log_DEBUG.info(self.word)
            Logger.log_DEBUG.info("词性为：")
            Logger.log_DEBUG.info(self.vocabulary)
            Logger.log_DEBUG.info("对应同义词为：")
            Logger.log_DEBUG.info(self.replace_sys)
            Logger.log_DEBUG.info("可替换词位置为：")
            Logger.log_DEBUG.info(self.replace_Loc)

            Logger.log_DEBUG.info("可替换主体同义词为：")
            Logger.log_DEBUG.info(self.indiv_sys)

        except Exception as e:
            s = "句子分词是出现异常Senten_Seg" + str(e)
            Logger.log_ERROR.error(s)
            Logger.log_ERROR.exception(sys.exc_info())
            raise TypeError(s)


if __name__ == '__main__':
    # 新建同义词获取类实例
    SD = Synonyms_Dictionary({}, {}, config_SR.sys_path, config_SR.sys_indiv_path)
    print(config_SR.userdict_path)
    jieba.load_userdict(config_SR.userdict_path)  # file_name为自定义词典的路径
    s = '贷记卡查询自扣还款账号'
    senten = Senten_Seg(s, SD)
    print("可替换位置：")
    print(senten.replace_Loc)
    print("可替换同义词：")
    print(senten.replace_sys)
    print("可替换主体同义词：")
    print(senten.indiv_sys)
