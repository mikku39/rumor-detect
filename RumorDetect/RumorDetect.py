from typing import List, Dict
from dotenv import load_dotenv
from RumorDetect.model import (
    BaseNewsModel,
    BaseSummaryModel,
    BaseCompareModel,
    BaseJudgeModel,
)
from RumorDetect.tools.data_tools import power_mean
from RumorDetect.component import get_keywords
from collections import Counter
import tabulate
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

    参数:

        auto_init : bool = True # 是否自动初始化各模块
        enable_keyword : bool = True # 是否开启关键词提取
        enable_summary : bool = True——是否开启概要生成
        enable_search_compare : bool = True——是否开启新闻搜索比较功能
        enable_judge : bool = True——是否开启模型谣言判断功能
        news_mode : List[str] = ["google"]——新闻搜索模式
        {
            "tjsx": 天聚数行API # 需要配置环境变量 TJSX_API_KEY
            "google": google api搜索 # 需要配置环境变量 CSE_ID 和 CSE_API_KEY
            "bing": bing api搜索    # 需要配置环境变量 BING_SEARCH_KEY
            "bing_spider": bing爬虫搜索 # 使用爬虫对bing进行搜索，由于各种限制，返回结果不是特别稳定
        }
        summary_mode : List[str] =["pegasus"]——摘要生成模式
        {
            "pegasus": 天马模型 # 会自动下载模型到 {$HOME}/tmp
            "baidu": 使用百度API进行概述 # 需要配置环境变量 BAIDU_API_KEY 和 BAIDU_API_SECRET
            "ernie_bot": 使用百度ernie大模型进行概述 # 需要配置环境变量 ERNIE_BOT_KEY
        }
        compare_mode : List[str] =["entailment"]——新闻比较模式
        {
            "entailment": 语义蕴含
            "match": 语义匹配
            "baidu": 使用百度API进行比较(最大支持512字节) # 需要配置环境变量 BAIDU_API_KEY 和 BAIDU_API_SECRET
            "ernie_bot": 使用百度ernie大模型进行比较 # 需要配置环境变量 ERNIE_BOT_KEY 和 ERNIE_BOT_SECRET
        }
        judge_mode : List[str] = ["cnn"]——谣言判断模式
        {
            "cnn": 使用CNN模型进行判断
            "ernie_bot": 使用百度ernie大模型进行判断
        }
        banned_url : List[str] = ["zhihu", "baijiahao"] # 在新闻搜索中禁止的网站
        keyword_limit_num : int = 8 # 关键词提取的最大数量
        news_limit_num : int = 5 # 新闻搜索的最大数量
    """

    def __init__(
        self,
        auto_init: bool = True,
        enable_keyword: bool = True,
        enable_summary: bool = True,
        enable_search_compare: bool = True,
        enable_judge: bool = True,
        news_mode: List[str] = ["google"],
        summary_mode: List[str] = ["pegasus"],
        compare_mode: List[str] = ["entailment"],
        judge_mode: List[str] = ["cnn"],
        banned_url: List[str] = ["zhihu", "baijiahao"],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
    ):
        load_dotenv()
        self.update_params(locals())
        self.initialized = False
        self.judge_initialized = False
        self.search_compare_initialized = False
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

        self.news_dict = {  # 多种新闻搜索方式
            "tjsx": TJSXNewsModel,  # 天聚数行API
            "google": GoogleNewsModel,  # google api搜索
            "bing": BingNewsModel,  # bing api搜索
            "bing_spider": BingSpiderNewsModel,  # bing爬虫搜索
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

    # 运行
    def run(self, sent):
        if not self.initialized:
            print("未初始化,正在按各模块的 mode 配置初始化")
            self.init()
        self.sent = sent

        if self.enable_search_compare:
            if len(sent) < 15 and self.enable_keyword:
                print("输入文本长度小于15，令自身为关键词即可")
                self.enable_keyword = False
                self.keywords = [self.sent]

            if self.enable_keyword:
                self.keywords = get_keywords(self.sent)

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
                        sent, self.news_list
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
        if self.enable_judge:
            self.judge_result = []
            for judge_model in self.judge_models:
                self.judge_result.append(judge_model.judge(self.sent))
            self.judge_result = self.aggregate_judge_result()  # 将所有模型的结果聚合
            print("judge结果如下：")
            print(tabulate.tabulate(self.judge_result, headers="keys", tablefmt="grid"))

    # Debug模式运行 Search_compare功能，以迭代器的方式执行每一步可以查看和修改中间变量
    def debug_run(self, sent):
        self.sent = sent
        if not self.enable_search_compare:
            print("Debug模式只支持新闻比较功能，当前该功能未开启")
            return
        if not self.initialized:
            print("未初始化,正在按各模块的 mode 配置初始化")
            self.init()

        if len(sent) < 15 and self.enable_keyword:
            print("输入文本长度小于15，令自身为关键词即可")
            self.enable_keyword = False
            self.keywords = [self.sent]

        if self.enable_keyword:
            self.keywords = get_keywords(self.sent)

        print(self.keywords)
        print(
            "关键词搜索完毕。查看或修改中间变量请使用函数 self.get_intermediate() 和 self.update_params(key, value)"
        )
        print(self.keywords)
        yield
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
        yield
        if self.enable_summary:
            self.sent, self.news_list = self.summary_models[0].get_summary(
                sent, self.news_list
            )
        print(
            "概要完毕.查看或修改中间变量请使用函数 self.get_intermediate() 和 self.update_params(key, value)"
        )
        print("再次运行即为最终结果")
        yield
        self.compare_result = []
        for compare_model in self.compare_models:
            self.compare_result.append(
                (compare_model.compare(self.sent, self.news_list))
            )
        self.compare_result = self.aggregate_compare_result()  # 将所有模型的结果聚合
        print("search_compare结果如下：")
        print(tabulate.tabulate(self.compare_result, headers="keys", tablefmt="grid"))
        print("比较完毕.")
        return

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
                    power_mean([value["score"] for value in values]) > 0.5
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

    def list_available_news_mode(self):
        return self.news_dict.keys()

    def list_available_summary_mode(self):
        return self.summary_dict.keys()

    def list_available_compare_mode(self):
        return self.compare_dict.keys()

    def list_available_judge_mode(self):
        return self.judge_dict.keys()

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

    def get_news_mode(self):
        return self.news_mode

    def get_summary_mode(self):
        return self.summary_mode

    def get_compare_mode(self):
        return self.compare_mode

    def get_judge_mode(self):
        return self.judge_mode
