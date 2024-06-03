import os
from typing import List, Dict
from dotenv import load_dotenv
from RumorDetect.model import (
    BaseKeywordsModel,
    BaseNewsModel,
    BaseSummaryModel,
    BaseCompareModel,
    BaseJudgeModel,
)
from RumorDetect.tools.data_tools import power_mean
from collections import Counter
import tabulate
from RumorDetect.modules.keywords_module import JiebaKeywordsModule
from RumorDetect.modules.news_module import (
    TJSXNewsModel,
    GoogleNewsModel,
    BingNewsModel,
    BingSpiderNewsModel,
)
from RumorDetect.modules.summary_module import (
    PegasusSummaryModel,
    BaiduSummaryModel,
    ErnieBotSummaryModel,
)
from RumorDetect.modules.compare_module import (
    MatchingCompareModel,
    EntailmentCompareModel,
    BaiduCompareModel,
    ErnieBotCompareModel,
)
from RumorDetect.modules.judge_module import (
    CNNJudgeModel,
    ErnieBotJudgeModel,
)


class rumor_detect:
    """
    这是这个包的核心类，用于进行谣言检测。
    在初始化时，可以根据配置的参数选择是否自动初始化各模块，以及是否开启关键词提取、概要生成、新闻比较、谣言判断等功能。
    下面是各个参数的说明：

    Args:
        auto_init : bool 是否自动初始化各模块
        enable_keyword : bool 是否开启关键词提取
        enable_summary : bool 是否开启概要生成
        enable_search_compare : bool 是否开启新闻搜索比较功能
        enable_judge : bool 是否开启模型谣言判断功能
        news_mode : List[str] 新闻搜索模式 - google / bing / bing_spider / tjsx
        summary_mode : List[str] 摘要生成模式 - pegasus / baidu / ernie_bot
        compare_mode : List[str] 新闻比较模式 - entailment / match / baidu / ernie_bot
        judge_mode : List[str] 谣言判断模式 - cnn / ernie_bot
        banned_url : List[str] 在新闻搜索中禁止的网站 
        keyword_limit_num : int 关键词提取的最大数量
        news_limit_num : int 新闻搜索的最大数量
    """

    def __init__(
        self,
        auto_init: bool = True,
        enable_keyword: bool = True,
        enable_summary: bool = True,
        enable_search_compare: bool = True,
        enable_judge: bool = True,
        keywords_mode: List[str] = ["jieba"],
        news_mode: List[str] = ["bing"],
        summary_mode: List[str] = ["ernie_bot"],
        compare_mode: List[str] = ["entailment","ernie_bot"],
        judge_mode: List[str] = ["cnn","ernie_bot"],
        banned_url: List[str] = ["zhihu", "baijiahao"],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
    ):
        load_dotenv(os.path.join(os.getcwd(), '.env'))
        self.update_params(locals())
        self.initialized = False
        self.judge_initialized = False
        self.search_compare_initialized = False
        self.keywords_models = None
        self.news_models = None
        self.summary_models = None
        self.compare_models = None
        self.judge_models = None
        self.keywords = []
        self.news_list = []
        self.sent = ""
        self.compare_result = []
        self.judge_result = []
        self.lab = ["谣言", "非谣言"]

        if len(self.summary_mode) > 1:
            print("概要模块只支持单一模式，已自动选择第一个模式")
            self.summary_mode = [self.summary_mode[0]]
            
        self.keywords_dict = {
            "jieba": JiebaKeywordsModule,  # 使用jieba提取关键词
        }

        self.news_dict = {  # 多种新闻搜索方式
            "google": GoogleNewsModel,  # google api搜索
            "bing": BingNewsModel,  # bing api搜索
            "bing_spider": BingSpiderNewsModel,  # bing爬虫搜索
            "tjsx": TJSXNewsModel,  # 天聚数行API
        }

        self.summary_dict = {  # 多种摘要方式
            "pegasus": PegasusSummaryModel,  # 天马模型
            "baidu": BaiduSummaryModel,  # 使用百度API进行概述
            "ernie_bot": ErnieBotSummaryModel,  # 使用百度ernie大模型进行概述
        }

        self.compare_dict = {  # 多种新闻比较方式
            "entailment": EntailmentCompareModel,  # 语义蕴含
            "match": MatchingCompareModel,  # 语义匹配
            "baidu": BaiduCompareModel,  # 使用百度API进行比较(最大支持512字节)
            "ernie_bot": ErnieBotCompareModel,  # 使用百度ernie大模型进行比较
        }

        self.judge_dict = {  # 多种模型判断方式
            "cnn": CNNJudgeModel,  # CNN模型
            "ernie_bot": ErnieBotJudgeModel,  # 使用百度ernie大模型进行判断
        }

        if self.auto_init:
            self.init()

    def update_params(self, params: Dict[str, str]):
        for key, value in params.items():
            if key != "self":
                self.__setattr__(key, value)

    # 初始化各模块
    def init(self):
        # 搜索相关新闻并比较的初始化
        if self.enable_search_compare and not self.search_compare_initialized:
            self.keywords_init()
            self.news_init()
            if self.enable_summary:
                self.summary_init()
            self.compare_init()
            self.search_compare_initialized = True
        # 只使用算法模型进行judge是否是谣言的初始化
        if self.enable_judge and not self.judge_initialized:
            self.judge_init()
            self.judge_initialized = True

        self.initialized = True


    def run(self, sent):
        '''
            运行谣言检测模型
        '''
        if not self.initialized or not self.search_compare_initialized or not self.judge_initialized:
            print("未初始化,正在按各模块的 mode 配置初始化")
            self.init()
        self.sent = sent
        self.final_score = []
        self.keywords = []
        if self.enable_search_compare:
            if self.enable_keyword:
                for keywords_model in self.keywords_models:
                    self.keywords.extend(keywords_model.get_keywords(self.sent))
            else:
                self.keywords = [self.sent]
            print(f"关键字为{self.keywords}")
            self.news_list = []
            for news_model in self.news_models:
                self.news_list.extend(
                    news_model.find_news(
                        self.keywords,
                        self.keyword_limit_num,
                        self.news_limit_num,
                        self.banned_url,
                    )
                )
            if not self.news_list:
                print("未找到相关新闻")
            else:
                if self.enable_summary:
                    self.sent, self.news_list = self.summary_models[0].get_summary(
                        self.sent, self.news_list
                    )
                print(self.news_list)
                self.compare_result = []
                for compare_model in self.compare_models:
                    self.compare_result.append(
                        (compare_model.compare(self.sent, self.news_list))
                    )
                self.compare_result = (
                    self.aggregate_compare_result()
                )  # 将所有模型的结果聚合
                print("search_compare结果如下：")
                print(
                    tabulate.tabulate(
                        self.compare_result, headers="keys", tablefmt="grid"
                    )
                )
                self.final_score.extend([compare_result["final_score"] for compare_result in self.compare_result])
        if self.enable_judge:
            self.judge_result = []
            for judge_model in self.judge_models:
                self.judge_result.append(judge_model.judge(self.sent))
            self.judge_result = self.aggregate_judge_result()  # 将所有模型的结果聚合
            print("judge结果如下：")
            print(tabulate.tabulate(self.judge_result, headers="keys", tablefmt="grid"))
            self.final_score.extend([judge_result["finalscore"] for judge_result in self.judge_result])
        final_score = power_mean(self.final_score)
        print(f"最终置信度为{final_score},结果为{self.lab[final_score > 0.6]}")
        return {"score":final_score, "result":self.lab[final_score > 0.6]}

    def debug_run(self, sent):
        '''
            Debug模式运行 Search_compare功能，以迭代器的方式执行每一步可以查看和修改中间变量
        '''
        self.sent = sent
        self.final_score = []
        self.keywords = []
        if not self.enable_search_compare:
            print("Debug模式只支持新闻比较功能，当前该功能未开启")
            return
        if not self.initialized:
            print("未初始化,正在按各模块的 mode 配置初始化")
            self.init()
            
        if self.enable_keyword:
            for keywords_model in self.keywords_models:
                self.keywords.extend(keywords_model.get_keywords(self.sent))
        else:
            self.keywords = [self.sent]
        print(
            "关键词搜索完毕。查看或修改中间变量请使用函数 self.get_intermediate() 和 self.update_params(key, value)"
        )
        yield {"result":"功能未结束"}
        self.news_list = []
        for news_model in self.news_models:
            self.news_list.extend(
                news_model.find_news(
                    self.keywords,
                    self.keyword_limit_num,
                    self.news_limit_num,
                    self.banned_url,
                )
            )
        print(
            "新闻搜索完毕.查看或修改中间变量请使用函数 self.get_intermediate() 和 self.update_params(key, value)"
        )
        yield {"result":"功能未结束"}
        if self.enable_summary:
            self.sent, self.news_list = self.summary_models[0].get_summary(
                self.sent, self.news_list
            )
        print(
            "概要完毕.查看或修改中间变量请使用函数 self.get_intermediate() 和 self.update_params(key, value)"
        )
        print("再次运行即为最终结果")
        yield {"result":"功能未结束"}
        self.compare_result = []
        for compare_model in self.compare_models:
            self.compare_result.append(
                (compare_model.compare(self.sent, self.news_list))
            )
        self.compare_result = self.aggregate_compare_result()  # 将所有模型的结果聚合
        print("search_compare结果如下：")
        print(tabulate.tabulate(self.compare_result, headers="keys", tablefmt="grid"))
        print("比较完毕.")
        self.final_score = [compare_result["final_score"] for compare_result in self.compare_result]
        final_score = power_mean(self.final_score)
        print(f"最终置信度为{final_score},结果为{self.lab[final_score > 0.6]}")
        
        yield {"score":final_score, "result":self.lab[final_score > 0.6]}

    # 聚合 search_compare 功能中多个 compare 函数的结果
    def aggregate_compare_result(self):
        result = [
            {
                "source": values[0]["source"],
                "news": values[0]["news"],
                "news_url": values[0]["news_url"],
                "scores": dict(
                    zip(self.compare_mode, [value["score"] for value in values])
                ),
                "final_score": power_mean([value["score"] for value in values]),
                "predict": self.lab[
                    power_mean([value["score"] for value in values]) > 0.6
                ],
            }
            for values in zip(*self.compare_result)
        ]
        return result

    # 聚合 judge 功能中多个 judge 函数的结果
    def aggregate_judge_result(self):
        result = [
            {
                "source": self.judge_result[0]["source"],
                "scores": dict(
                    zip(self.judge_mode, [x["score"] for x in self.judge_result])
                ),
                "finalscore": power_mean([x["score"] for x in self.judge_result]),
                "predict": self.lab[
                    power_mean([x["score"] for x in self.judge_result]) > 0.6
                ],
            }
        ]
        return result

    # 获取中间变量
    def get_intermediate(self):
        return {
            "sent": self.sent,
            "news_list": self.news_list,
            "keywords": self.keywords,
        }
    
    def keywords_init(self):
        self.keywords_models = [self.keywords_dict[key]() for key in self.keywords_mode]
        
    def news_init(self):
        self.news_models = [self.news_dict[news_mode]() for news_mode in self.news_mode]

    def summary_init(self):
        self.summary_models = [self.summary_dict[self.summary_mode[0]]()]

    def compare_init(self):
        self.compare_models = [
            self.compare_dict[compare_mode]() for compare_mode in self.compare_mode
        ]

    def judge_init(self):
        self.judge_models = [
            self.judge_dict[judge_mode]() for judge_mode in self.judge_mode
        ]

    def list_available_keywords_mode(self):
        return list(self.keywords_dict.keys())

    def list_available_news_mode(self):
        return list(self.news_dict.keys())

    def list_available_summary_mode(self):
        return list(self.summary_dict.keys())

    def list_available_compare_mode(self):
        return list(self.compare_dict.keys())

    def list_available_judge_mode(self):
        return list(self.judge_dict.keys())

    def add_keywords_mode(self, key, value):
        if issubclass(value, BaseKeywordsModel):
            self.keywords_dict[key] = value
        else:
            print("请传入正确的关键词模型，必须是BaseKeywordsModel的子类")

    def add_news_mode(self, key, value):
        if issubclass(value, BaseNewsModel):
            self.news_dict[key] = value
        else:
            print("请传入正确的新闻模型，必须是BaseNewsModel的子类")

    def add_summary_mode(self, key, value):
        if issubclass(value, BaseSummaryModel):
            self.summary_dict[key] = value
        else:
            print("请传入正确的概述模型，必须是BaseSummaryModel的子类")

    def add_compare_mode(self, key, value):
        if issubclass(value, BaseCompareModel):
            self.compare_dict[key] = value
        else:
            print("请传入正确的比较模型，必须是BaseCompareModel的子类")

    def add_judge_mode(self, key, value):
        if issubclass(value, BaseJudgeModel):
            self.judge_dict[key] = value
        else:
            print("请传入正确的判断模型，必须是BaseJudgeModel的子类")

    def set_keywords_mode(self, mode: List[str]):
        if Counter(mode) != Counter(self.keywords_mode):
            self.search_compare_initialized = False
        self.keywords_mode = mode
    
    def set_news_mode(self, mode: List[str]):
        if Counter(mode) != Counter(self.news_mode):
            self.search_compare_initialized = False
        self.news_mode = mode

    def set_summary_mode(self, mode: List[str]):
        if Counter(mode) != Counter(self.summary_mode):
            self.search_compare_initialized = False
        self.summary_mode = mode

    def set_compare_mode(self, mode: List[str]):
        if Counter(mode) != self.compare_mode:
            self.search_compare_initialized = False
        self.compare_mode = mode

    def set_judge_mode(self, mode: List[str]):
        if Counter(mode) != Counter(self.judge_mode):
            self.judge_initialized = False
        self.judge_mode = mode
        
    def get_keywords_mode(self):
        return self.keywords_mode

    def get_news_mode(self):
        return self.news_mode

    def get_summary_mode(self):
        return self.summary_mode

    def get_compare_mode(self):
        return self.compare_mode

    def get_judge_mode(self):
        return self.judge_mode
