from typing import List
from RumorDetect.model import BaseNewsModel
from RumorDetect.tools.data_tools import bing_search, bing_spider_search, get_news_list, google_search, tx_search,data_ban_url

class TJSXNewsModel(BaseNewsModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        pass

    def find_news(
        self,
        keyword_list: List[str],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
        banned_url: List[str] = [],
    ):
        '''
            根据关键字列表通过天行数据接口查找新闻
        '''
        if len(keyword_list) > keyword_limit_num:
            keyword_list = keyword_list[:keyword_limit_num]
        tx_data = tx_search(keyword_list)
        if tx_data["code"] != 200:
            return []
        tx_data = tx_data["result"]["newslist"]
        tx_data = data_ban_url(tx_data, banned_url, news_limit_num)
        if len(tx_data) > news_limit_num:
            tx_data = tx_data[:news_limit_num]
        return get_news_list(tx_data)
    
class GoogleNewsModel(BaseNewsModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        pass

    def find_news(
        self,
        keyword_list: List[str],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
        banned_url: List[str] = [],
    ):
        '''
            根据关键字列表通过 Google 接口查找新闻
        '''
        if len(keyword_list) > keyword_limit_num:
            keyword_list = keyword_list[:keyword_limit_num]
        keyword_str = " ".join(keyword_list)
        google_data = google_search(keyword_str)
        for data in google_data:
            data["url"] = data["link"]
        google_data = data_ban_url(google_data, banned_url, news_limit_num)
        if len(google_data) > news_limit_num:
            google_data = google_data[:news_limit_num]
        return get_news_list(google_data)


class BingNewsModel(BaseNewsModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        pass
    
    def find_news(
        self,
        keyword_list: List[str],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
        banned_url: List[str] = [],
    ):
        '''
            根据关键字列表通过 Bing 接口查找新闻
        '''
        if len(keyword_list) > keyword_limit_num:
            keyword_list = keyword_list[:keyword_limit_num]
        keyword_str = " ".join(keyword_list)
        bing_data = bing_search(keyword_str)
        for data in bing_data:
            data["title"] = data["name"]
        bing_data = data_ban_url(bing_data, banned_url, news_limit_num)
        if len(bing_data) > news_limit_num:
            bing_data = bing_data[:news_limit_num]
        return get_news_list(bing_data)
    
class BingSpiderNewsModel(BaseNewsModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        pass
    
    def find_news(
        self,
        keyword_list: List[str],
        keyword_limit_num: int = 8,
        news_limit_num: int = 5,
        banned_url: List[str] = [],
    ):
        '''
        根据关键字列表通过 Bing 爬虫接口查找新闻
        Args:
            keyword_list: 
            keyword_limit_num:  
            news_limit_num:
            banned_url: 
        '''
        if len(keyword_list) > keyword_limit_num:
            keyword_list = keyword_list[:keyword_limit_num]
        keyword_str = " ".join(keyword_list)
        bing_data = bing_spider_search(keyword_str)
        for data in bing_data:
            data["title"] = data["name"]
        bing_data = data_ban_url(bing_data, banned_url, news_limit_num)
        if len(bing_data) > news_limit_num:
            bing_data = bing_data[:news_limit_num]
        return get_news_list(bing_data)