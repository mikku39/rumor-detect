from dataclasses import dataclass
from typing import Callable, List
from RumorDetect.component import (
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
from RumorDetect.tools.model_tools import cnn_init, entailment_init, match_init, pegasus_init
from RumorDetect.tools.data_tools import noop_init
from collections import Counter
import tabulate
@dataclass
class model_template:
    init: Callable
    infer: Callable


class rumor_detect:
    def __init__(
        self,
        auto_init=True,
        enable_keyword=True,
        enable_summary=True,
        enable_search_compare=True,
        enable_judge=True,
        news_mode=["google"],
        summary_mode=["pegasus"],
        compare_mode=["entailment"],
        judge_mode=["cnn"],
        keyword_limit_num=8,
        news_limit_num=5,
        banned_url=["zhihu", "baijiahao"],
    ):
        load_dotenv()
        self.enable_summary = enable_summary
        self.enable_keyword = enable_keyword
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
        self.summary_models = None
        self.compare_models = None
        self.judge_models = None
        self.keyword = []
        self.news_list = []
        self.sent = ""
        self.compare_result = []
        self.judge_result = []
        self.banned_url = banned_url
        self.lab = ["谣言", "非谣言"]

        if len(self.summary_mode) > 1:
            print("概要模块只支持单一模式，已自动选择第一个模式")
            self.summary_mode = [self.summary_mode[0]]

        self.find_news_dict = {  # 多种新闻搜索方式
            "tianxing": tianxing_find_news,  # 天聚数行API
            "google": google_find_news,  # google搜索
            "bing": bing_find_news,  # bing搜索
        }

        self.summary_dict = {  # 多种摘要方式
            "pegasus": model_template(
                init=pegasus_init, infer=pegasus_group_infer
            ),  # 天马模型
            "baidu": model_template(
                init=noop_init, infer=baidu_summary_group_infer
            ),  # 使用百度API进行总结
        }

        self.compare_dict = {  # 多种新闻比较方式
            "entailment": model_template(
                init=entailment_init, infer=check_entailment
            ),  # 语义蕴含
            "match": model_template(init=match_init, infer=check_match),  # 语义匹配
            "baidu": model_template(
                init=noop_init,
                infer=baidu_compare,
            ),  # 使用百度API进行比较(最大支持512字节)
        }

        self.judge_dict = {  # 多种模型判断方式
            "cnn": model_template(init=cnn_init, infer=cnn_infer),  # CNN模型
        }

        if self.auto_init:
            self.init()

    def init(self):
        # 搜索相关新闻并比较的初始化
        if self.enable_search_compare and not self.search_compare_initialized:
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
            for news_mode in self.news_mode:
                self.news_list.extend(
                    self.find_news_dict[news_mode](
                        self.keywords, self.keyword_limit_num, self.news_limit_num, self.banned_url
                    )
                )
            if not self.news_list:
                print("未找到相关新闻")
            else:
                if self.enable_summary:
                    self.sent, self.news_list = self.summary_dict[
                        self.summary_mode[0]
                    ].infer(self.summary_models[0], sent, self.news_list)
                print(self.news_list)
                self.compare_result = []
                for idx, compare_mode in enumerate(self.compare_mode):
                    self.compare_result.append(
                        (
                            self.compare_dict[compare_mode].infer(
                                self.compare_models[idx], self.sent, self.news_list
                            )
                        )
                    )
                self.compare_result = [
                    {
                        "source": values[0]["source"],
                        "news": values[0]["news"],
                        "news_url": values[0]["news_url"],
                        "score": sum([x["score"] for x in values]) / len(values),
                        "predict": self.lab[
                            (sum([x["score"] for x in values]) / len(values)) > 0.5
                        ],
                    }
                    for values in zip(*self.compare_result)
                ]
                print("search_compare结果如下：")
                print(tabulate.tabulate(self.compare_result, headers="keys", tablefmt="grid"))
        if self.enable_judge:
            self.judge_result = []
            for idx, judge_mode in enumerate(self.judge_mode):
                self.judge_result.append(self.judge_dict[judge_mode].infer(self.judge_models[idx], self.sent))
            self.judge_result = [{
                "source": self.judge_result[0]["source"],
                "score": sum([x["score"] for x in self.judge_result]) / len(
                    self.judge_result
                ),
                "predict": self.lab[
                    (sum([x["score"] for x in self.judge_result]) / len(self.judge_result))
                    > 0.6
                ],
            }]
            print("judge结果如下：")
            print(tabulate.tabulate(self.judge_result, headers="keys", tablefmt="grid"))

    def debug_run(self, sent):
        if not self.enable_search_compare:
            print("Debug模式只支持新闻比较功能，当前该功能未开启")
            return
        self.keywords = get_keywords(self.sent)
        print(
            "关键词搜索完毕。查看或修改中间变量请使用函数 self.get_intermediate() 和 self.set_intermediate(key, value)"
        )
        yield
        self.news_list = self.find_news_dict[self.news_mode](
            self.keywords, self.keyword_limit_num, self.news_limit_num
        )
        print(
            "新闻搜索完毕.查看或修改中间变量请使用函数 self.get_intermediate() 和 self.set_intermediate(key, value)"
        )
        yield
        if self.enable_summary:
            self.sent, self.news_list = self.summary_dict[self.summary_mode].infer(
                self.summary_model, sent, self.news_list
            )
            print(
                "概要完毕.查看或修改中间变量请使用函数 self.get_intermediate() 和 self.set_intermediate(key, value)"
            )
            yield
        self.compare_dict[self.compare_mode].infer(
            self.compare_model, self.sent, self.news_list
        )
        print("比较完毕.查看或修改中间变量请使用函数 self.get_intermediate()")
        return

    def get_intermediate(self):
        return {
            "sent": self.sent,
            "news_list": self.news_list,
            "keywords": self.keywords,
        }

    def set_intermediate(self, key, value):
        self.__setattr__(key, value)

    def summary_init(self):
        self.summary_models = [self.summary_dict[self.summary_mode[0]].init()]

    def compare_init(self):
        self.compare_models = [
            self.compare_dict[compare_mode].init() for compare_mode in self.compare_mode
        ]

    def judge_init(self):
        self.judge_models = [
            self.judge_dict[judge_mode].init() for judge_mode in self.judge_mode
        ]

    def list_available_news_mode(self):
        return self.find_news_dict.keys()

    def list_available_summary_mode(self):
        return self.summary_dict.keys()

    def list_available_compare_mode(self):
        return self.compare_dict.keys()

    def list_available_judge_mode(self):
        return self.judge_dict.keys()

    def add_news_mode(self, key, value):
        self.find_news_dict[key] = value

    def add_summary_mode(self, key, value):
        self.summary_dict[key] = value

    def add_compare_mode(self, key, value):
        self.compare_dict[key] = value

    def add_judge_mode(self, key, value):
        self.judge_dict[key] = value

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


if __name__ == "__main__":
    # instance = rumor_detect(enable_summary= False)
    # instance = rumor_detect(news_mode="tianxing")
    instance = rumor_detect(
        news_mode=["bing"],
        summary_mode=["baidu"],
        compare_mode=["baidu", "match"],
        judge_mode=["cnn"],
    )
    # sent = "华晨宇有小孩子了"
    # sent = "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例"
    sent = "狂飙兄弟在连云港遇到鬼秤"
    instance.run(sent)
