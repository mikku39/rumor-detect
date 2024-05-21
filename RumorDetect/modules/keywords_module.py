from typing import List
from RumorDetect.model import BaseKeywordsModel
import jieba.analyse

class JiebaKeywordsModule(BaseKeywordsModel):
    '''
        使用Jieba提供的TF-IDF方法提取关键词
    '''
    def init(self):
        pass
    
    def get_keywords(self, sent: str, keyword_limit_num: int = 8) -> List[str]:
        if len(sent) < 15:
            print("输入文本长度小于15，令自身为关键词即可")
            return [sent]
        return jieba.analyse.extract_tags(sent, topK=keyword_limit_num)