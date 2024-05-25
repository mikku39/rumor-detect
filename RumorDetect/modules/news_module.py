from typing import Dict, List
from RumorDetect.model import BaseNewsModel
from RumorDetect.tools.data_tools import bing_search, bing_spider_search, get_news_list, google_search, tx_search,data_ban_url

class TJSXNewsModel(BaseNewsModel):
    '''
        根据天聚数行新闻接口查找新闻。该接口查找到的新闻大致在 china.news站点下。
        需要注意的是该接口仅会根据第一个关键字进行查找，且返回的新闻数量可能不足 news_limit_num。
        需要配置环境变量TJSX_API_KEY，获取方式见：https://www.tianapi.com/console/
    '''
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
    ) -> List[Dict[str, str]]:
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
    '''
        根据 Google 搜索接口查找新闻。
        需要配置环境变量：CSE_API_KEY 和 CSE_ID
        API KEY 获取方式见：https://programmablesearchengine.google.com/u/1/controlpanel/all
    '''
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
    ) -> List[Dict[str, str]]:
        if len(keyword_list) > keyword_limit_num:
            keyword_list = keyword_list[:keyword_limit_num]
        keyword_str = " ".join(keyword_list)
        google_data = google_search(keyword_str, banned_url)
        for data in google_data:
            data["url"] = data["link"]
        google_data = data_ban_url(google_data, banned_url, news_limit_num)
        if len(google_data) > news_limit_num:
            google_data = google_data[:news_limit_num]
        return get_news_list(google_data)


class BingNewsModel(BaseNewsModel):
    '''
        根据 Bing 搜索接口查找新闻。
        需要配置环境变量：BING_SEARCH_KEY
        API KEY 获取方式见：https://portal.azure.com/
    '''
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
    ) -> List[Dict[str, str]]:
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
    '''
        使用爬虫，直接向bing发送request请求，获取结果
        不过由于反爬虫机制，很多时候会被拦截，所以不推荐使用
    '''
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
    ) -> List[Dict[str, str]]:
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