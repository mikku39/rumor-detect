from typing import Dict, List, Tuple


class BaseNewsModel:
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
        raise NotImplementedError()

class BaseSummaryModel:
    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def get_summary(self, sent : str, news_list:List[Tuple]) -> Tuple[str, List[Tuple]]: 
        raise NotImplementedError()

class BaseCompareModel:
    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def compare(self, sent : str, news_list:List[Tuple]) -> List[Dict]: 
        raise NotImplementedError()

class BaseJudgeModel:
    def __init__(self) -> None:
        self.init()

    def init(self):
        raise NotImplementedError()

    def judge(self, sent : str) -> Dict: 
        raise NotImplementedError()