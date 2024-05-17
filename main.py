from dataclasses import dataclass
from typing import Callable
from component import (
    baidu_compare,
    baidu_summary_group_infer,
    get_keywords,
    pegasus_group_infer,
    check_match,
    cnn_infer,
    check_entailment,
    tianxing_find_news,
    google_find_news,
    bing_find_news,
)
from dotenv import load_dotenv
from tools.model_tools import cnn_init, entailment_init, match_init, pegasus_init
from tools.data_tools import noop_init

@dataclass
class model_template:
    init: Callable
    infer: Callable


class rumor_detect:
    def __init__(
        self,
        auto_init=True,
        enable_summary=True,
        enable_search_compare=True,
        enable_judge=True,
        news_mode="google",
        summary_mode="pegasus",
        compare_mode="entailment",
        judge_mode="cnn",
        keyword_limit_num=8,
        news_limit_num=5,
    ):
        load_dotenv()
        self.enable_summary = enable_summary
        self.news_mode = news_mode
        self.summary_mode = summary_mode
        self.compare_mode = compare_mode
        self.judge_mode = judge_mode
        self.keyword_limit_num = keyword_limit_num
        self.news_limit_num = news_limit_num
        self.auto_init = auto_init
        self.initialized = False
        self.judge_initialized = False
        self.search_compare_initialized = False
        self.enable_search_compare = enable_search_compare
        self.enable_judge = enable_judge
        self.summary_model = None
        self.compare_model = None
        self.judge_model = None
        
        self.find_news_dict = {  # 多种新闻搜索方式
            "tianxing": tianxing_find_news,  # 天聚数行API
            "google": google_find_news,  # google搜索
            "bing": bing_find_news,  # bing搜索
        }

        self.summary_dict = {   # 多种摘要方式
            "pegasus": model_template( 
                init=pegasus_init, infer=pegasus_group_infer
            ),#天马模型
            "baidu": model_template( 
                init=noop_init, infer=baidu_summary_group_infer
            ),#使用百度API进行总结
        }

        self.compare_dict = {  # 多种新闻比较方式
            "entailment": model_template(
                init=entailment_init, infer=check_entailment
            ),  # 语义蕴含
            "match": model_template(
                init=match_init, infer=check_match
            ),  # 语义匹配
            "baidu": model_template(
                init=noop_init, infer=baidu_compare,
            ),  # 使用百度API进行比较(最大支持512字节)
        }

        self.judge_dict = {  # 多种模型判断方式
            "cnn": model_template(init=cnn_init, infer=cnn_infer),#CNN模型
        }

        if self.auto_init:
            self.init()

    def init(self):
        #搜索相关新闻并比较的初始化
        if self.enable_search_compare and not self.search_compare_initialized:
            if self.enable_summary:
                self.summary_init()
            self.compare_init()
            self.search_compare_initialized = True
        #只使用算法模型进行judge是否是谣言的初始化
        if self.enable_judge and not self.judge_initialized:
            self.judge_init()
            self.judge_initialized = True
            
            
        self.initialized = True

    def run(self, sent):
        if not self.initialized:
            print("未初始化,正在按各模块的 mode 配置初始化")
            self.init()
        if self.enable_search_compare:
            keywords = get_keywords(sent)
            news_list = self.find_news_dict[self.news_mode](
                keywords, self.keyword_limit_num, self.news_limit_num
            )
            if not news_list:
                print("未找到相关新闻")
            else:
                if self.enable_summary:
                    sent, news_list = self.summary_dict[self.summary_mode].infer(
                        self.summary_model, sent, news_list
                    )
                print(news_list)
                self.compare_dict[self.compare_mode].infer(
                    self.compare_model, sent, news_list
                )
            print(keywords)
        if self.enable_judge:
            self.judge_dict[self.judge_mode].infer(self.judge_model, sent)

    def summary_init(self):
        self.summary_model = self.summary_dict[self.summary_mode].init()
    
    def compare_init(self):
        self.compare_model = self.compare_dict[self.compare_mode].init()
    
    def judge_init(self):
        self.judge_model = self.judge_dict[self.judge_mode].init()

    def list_summary_mode(self):
        return self.summary_dict.keys()

    def list_compare_mode(self):
        return self.compare_dict.keys()

    def list_judge_mode(self):
        return self.judge_dict.keys()

    def list_news_mode(self):
        return self.find_news_dict.keys()

    def add_summary_mode(self, key, value):
        self.summary_dict[key] = value

    def add_compare_mode(self, key, value):
        self.compare_dict[key] = value

    def add_judge_mode(self, key, value):
        self.judge_dict[key] = value

    def add_news_mode(self, key, value):
        self.find_news_dict[key] = value

    def set_summary_mode(self, mode):
        if mode != self.summary_mode:
            self.search_compare_initialized = False
        self.summary_mode = mode

    def set_compare_mode(self, mode):
        if mode != self.compare_mode:
            self.search_compare_initialized = False
        self.compare_mode = mode

    def set_judge_mode(self, mode):
        if mode != self.judge_mode:
            self.judge_initialized = False
        self.judge_mode = mode

    def set_news_mode(self, mode):
        self.news_mode = mode


if __name__ == "__main__":
    # instance = rumor_detect(enable_summary= False)
    # instance = rumor_detect(news_mode="tianxing")
    instance = rumor_detect(news_mode="bing",compare_mode="baidu")
    # sent = "华晨宇有小孩子了"
    # sent = "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例"
    sent = "狂飙兄弟在连云港遇到鬼秤"
    instance.run(sent)
