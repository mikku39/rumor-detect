from typing import Dict, List, Tuple
import http.client, json, urllib
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import requests
from bs4 import BeautifulSoup
import os
import url2io_client
from RumorDetect.component import get_default_path, get_env

# 获取数据


def load_data(sentence):
    # 读取数据字典
    with open(f"{get_default_path()}/dict.txt", "r", encoding="utf-8") as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = "<unk>"
        data.append(int(dict_txt[s]))
    return data


def tx_search(keyword_list):
    '''
        使用天行数据接口根据关键词搜索新闻，需要配置环境变量 TJSX_API_KEY
    '''
    conn = http.client.HTTPSConnection("apis.tianapi.com")  # 接口域名
    params = urllib.parse.urlencode(
        {"key": get_env("TJSX_API_KEY"), "word": keyword_list[0]}
    )
    headers = {"Content-type": "application/x-www-form-urlencoded"}
    conn.request("POST", "/generalnews/index", params, headers)
    tianapi = conn.getresponse()
    result = tianapi.read()
    data = result.decode("utf-8")
    tx_data = json.loads(data)
    return tx_data


# 返回google搜索结果
def google_search(search_term, banned_url,**kwargs):
    '''
        使用谷歌搜索接口根据关键词搜索新闻，需要配置环境变量 CSE_API_KEY 和 CSE_ID
    '''
    api_key = get_env("CSE_API_KEY")
    cse_id = get_env("CSE_ID")
    exclude_query = ' '.join(f'-site:{site}' for site in banned_url)
    full_query = f'{search_term} {exclude_query}'
    query_params = {"q": full_query, "key": api_key, "cx": cse_id}
    query_params.update(kwargs)
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1", params=query_params
    )
    google_data = response.json()
    if "items" in google_data:
        return google_data["items"]
    print(
        f"谷歌搜索查询失败。返回数据为：{google_data}，KEY为{api_key}, CSE_ID为{cse_id}"
    )
    return []


# 返回bing搜索结果
def bing_search(search_term, **kwargs):
    '''
        使用bing搜索接口根据关键词搜索新闻，需要配置环境变量 BING_SEARCH_KEY
    '''
    api_key = get_env("BING_SEARCH_KEY")
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    headers.update(kwargs)

    query_params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers=headers,
        params=query_params,
    )
    response.raise_for_status()
    bing_data = response.json()
    if "webPages" in bing_data and "value" in bing_data["webPages"]:
        return bing_data["webPages"]["value"]
    print(f"bing搜索查询失败。返回数据为：{bing_data}，KEY为{api_key}")
    return []


def bing_spider_search(search_term, **kwargs):
    '''
        使用bing爬虫搜索相关新闻
    '''
    url = "https://www.bing.com/search"
    params = {"q": search_term}
    response = requests.get(url, params=params)
    result_list = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # 查找所有的搜索结果条目
        results = soup.find_all("li", {"class": "b_algo"})

        # 遍历每个结果并提取标题和链接
        for result in results:
            result_list.append(
                {
                    "title": result.find("h2").text.strip(),
                    "url": result.find("a")["href"],
                }
            )
        return result_list
    else:
        print("bing_spider Failed to retrieve the content")
        return []


def get_url_ctx(url: str) -> Dict:
    '''
    使用url2io接口获取网页正文内容

    Args:
        url: 网页链接

    Returns:
        爬取的网页正文内容
    '''
    configuration = url2io_client.Configuration()
    configuration.host = "http://url2api.applinzi.com"
    configuration.api_key["token"] = get_env("URL2IO_KEY","")
    api_instance = url2io_client.URL2ArticleApi(url2io_client.ApiClient(configuration))
    fields = ["text"]
    try:
        # 网页结构智能解析 HTTP Get 接口
        api_response = api_instance.get_article(url, fields=fields)
        return {"code": 200, "result": api_response.text}
    except Exception as e:
        print(f"查找网页{url},出错：{e}")
        return {"code": 403, "err_msg": e}


def beauty_ctx(ctx)->str:
    '''
        使用 BeautifulSoup 库去除网页标签，美化网页内容

        Args:
            ctx: 爬取网页正文内容

        Returns:
            优化后的正文内容
    '''
    soup = BeautifulSoup(ctx, "lxml")
    return soup.get_text()


# 返回网页内容
def get_news_list(data_list: List[Dict]) -> List[Tuple]:
    '''
        根据每条新闻的url爬取新闻内容，并以此做下一步计算。如果爬不到就直接返回标题

        Args:
            data_list: 通过新闻搜索模块获取的原始新闻列表

        Returns:
            尝试爬取正文信息后的新闻列表
    '''
    news_list = []
    for data in data_list:
        news_data = get_url_ctx(data["url"])
        if news_data["code"] == 200:
            news_list.append(
                (data["title"], data["url"], beauty_ctx(news_data["result"]))
            )
        else:
            news_list.append((data["title"], data["url"], data["title"]))
    return news_list


def data2np(input):
    data = load_data(input)
    data_np = np.array(data)
    data_np = (
        np.array(
            np.pad(
                data_np,
                (0, max(0, 150 - len(data_np))),
                "constant",
                constant_values=4409,
            )
        )
        .astype("int64")
        .reshape(-1)
    )

    if len(data_np) > 150:
        data_np = data_np[:150]

    infer_np_doc = to_variable(data_np)
    return infer_np_doc


def generate_data_source(sent, data_list):
    return [{"query": sent, "title": data[2], "label": 0} for data in data_list]


def generate_data(sent, data_list):
    return [[data[2], sent] for data in data_list]


# 截断到最大字节限制
def truncate_bytes(string, max_bytes=512, encoding="utf-8"):
    byte_str = string.encode(encoding)
    if len(byte_str) <= max_bytes:
        return string
    truncated_str = byte_str[:max_bytes].decode(encoding, "ignore")
    return truncated_str


def ernie_bot_summary_single_infer(token, text):
    url = get_env("ERIBOT_URL") + token

    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": text[:1024],
                }
            ],
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1,
            "system": "【功能说明】 你是一个专为提供文本概述设计的人工智能助手。你的任务是仅从所给文本中提取信息，生成一个简短的概述。在处理过程中，不得搜索或引入任何超出原始文本范围的信息。  【操作要求】  仅从用户提供的文本中分析和提取信息。 生成的概述必须直接基于输入的文本，严禁使用任何非输入文本的信息。 概述不得超过50字，并且必须紧密关联原文内容。 【示例】 如果输入文本是：“全球变暖导致极端天气事件的频率和强度增加，科学家们呼吁更多的气候行动。” 你的输出应该是：“科学家因全球变暖导致极端天气增加呼吁气候行动。”",
            "disable_search": True,
            "enable_citation": False,
            "response_format": "text",
        }
    )
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        result = json.loads(response.text)["result"]
    except Exception as e:
        print(f"使用ErnieBot进行summary的时候出错，截取前50个字符：{e}")
        return text[:50]
    return result


##对数平均
def power_mean(values, p=3):
    return np.power(np.mean(np.power(values, p)), 1 / p)


def data_ban_url(
    data: List[Dict[str, str]], banned_url: List[str], news_limit_num: int
):
    """
    根据 banned_url 列表过滤 data 中的新闻
    """
    if len(data) < news_limit_num:
        return data
    for banned in banned_url:
        for new in data:
            if banned in new["url"]:
                data.remove(new)
            if len(data) < news_limit_num:
                return data
    return data
