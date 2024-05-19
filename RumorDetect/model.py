from typing import Dict, List, Tuple


class BaseNewsModel:
    """
    这是新闻搜索模块的基类。用户需要基于该基类实现自己的新闻搜索模块。
    该类的实现需要实现以下方法：
    1. **init()** -> None: 初始化方法，用于初始化模型。
    2. **find_news**(
        self,
        keyword_list: List[str],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
        banned_url: List[str] = []
    ) -> List[Dict[str, str]]:
    """

    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def find_news(
        self,
        keyword_list: List[str],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
        banned_url: List[str] = [],
    ) -> List[Dict[str, str]]:
        """
        该方法用于搜索新闻。

        Args:
            keyword_list: 关键词列表。
            keyword_limit_num: 关键词数量限制。
            news_limit_num: 新闻数量限制。
            banned_url: 禁止的url列表。

        Returns:
            返回新闻列表[("title", "url", "content")]
        """
        raise NotImplementedError()


class BaseSummaryModel:
    """
    这是新闻概要模块的基类。用户需要基于该基类实现自己的新闻概要模块。
    该类的实现需要实现以下方法：
    1. **init()** -> None: 初始化方法，用于初始化模型。
    2. **get_summary**(self, sent : str, news_list:List[Tuple]) -> Tuple[str, List[Tuple]]
    """

    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def get_summary(self, sent: str, news_list: List[Tuple]) -> Tuple[str, List[Tuple]]:
        """
        该方法用于新闻概要。

        Args:
            sent: 原微博文本
            news_list: 原新闻列表

        Returns:
            返回一个 Tuple，第一个元素是原文本的 summary ，第二个元素是逐一 summary 后的新闻列表
        """
        raise NotImplementedError()


class BaseCompareModel:
    """
    这是新闻文本比较模块的基类。用户需要基于该基类实现自己的文本比较模块。
    该类的实现需要实现以下方法：
    1. **init()** -> None: 初始化方法，用于初始化模型。
    2. **compare**(self, sent : str, news_list:List[Tuple]) -> List[Dict]
    """

    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def compare(self, sent: str, news_list: List[Tuple]) -> List[Dict]:
        """
        该方法用于新闻文本比较。

        Args:
            sent: 微博文本
            news_list: 新闻列表

        Returns:
            返回一个列表。[{"source":原微博文本,"news": 新闻,"news_url": 新闻url,"score" :置信度，越大越不是谣言,"predict"：是否是谣言}]
        """
        raise NotImplementedError()


class BaseJudgeModel:
    """
    这是模型判断模块的基类。用户需要基于该基类实现自己的模型判断模块。
    该类的实现需要实现以下方法：
    1. **init()** -> None: 初始化方法，用于初始化模型。
    2. **judge**(self, sent : str) -> Dict
    """
    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def judge(self, sent: str) -> Dict:
        """
        该方法用于模型直接判断是否是谣言。

        Args:
            sent: 微博文本

        Returns:
            返回一个字典。[{"source":原微博文本,"score" :置信度，越大越不是谣言,"predict"：是否是谣言}]
        """
        raise NotImplementedError()
