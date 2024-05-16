import http.client, json, urllib
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import requests
from bs4 import BeautifulSoup
import os
# 获取数据


def load_data(sentence):
    # 读取数据字典
    with open("data/dict.txt", "r", encoding="utf-8") as f_data:
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
    conn = http.client.HTTPSConnection("apis.tianapi.com")  # 接口域名
    params = urllib.parse.urlencode(
        {"key": os.environ.get("TX_KEY"), "word": keyword_list[0]}
    )
    headers = {"Content-type": "application/x-www-form-urlencoded"}
    conn.request("POST", "/generalnews/index", params, headers)
    tianapi = conn.getresponse()
    result = tianapi.read()
    data = result.decode("utf-8")
    tx_data = json.loads(data)
    return tx_data

# 返回google搜索结果
def google_search(
    search_term,
    api_key=os.environ.get("CSE_API_KEY"),
    cse_id=os.environ.get("CSE_ID"),
    **kwargs
):
    query_params = {"q": search_term, "key": api_key, "cx": cse_id}
    query_params.update(kwargs)
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1", params=query_params
    )
    return response.json()["items"]


def get_url_ctx(url):
    conn = http.client.HTTPSConnection("apis.tianapi.com")  # 接口域名
    params = urllib.parse.urlencode(
        {"key": os.environ.get("TX_KEY"), "url": url}
    )
    headers = {"Content-type": "application/x-www-form-urlencoded"}
    conn.request("POST", "/htmltext/index", params, headers)
    tianapi = conn.getresponse()
    result = tianapi.read()
    data = result.decode("utf-8")
    news_data = json.loads(data)
    return news_data

def beauty_ctx(ctx):
    soup = BeautifulSoup(ctx, 'lxml')
    return soup.get_text()

# 返回网页内容
def get_news_list(data_list):
    news_list = []
    for data in data_list:
        news_data = get_url_ctx(data["url"])
        if news_data["code"] == 200:
            news_list.append(
                (data["title"], data["url"], beauty_ctx(news_data["result"]["content"]))
            )
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
    return [{"query" : sent, "title" : data[2], "label" : 0} for data in data_list]
    # return [
    #     {"query": sent, "title": data_list[0], "label": 0},
    #     {"query": sent, "title": data_list[0], "label": 0},
    #     {
    #         "query": "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例",
    #         "title": data_list[0],
    #         "label": 0,
    #     },
    #     {
    #         "query": "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例",
    #         "title": data_list[0],
    #         "label": 0,
    #     },
    #     {"query": sent, "title": "平均每人每天接触到4篇虚假新闻", "label": 0},
    #     {
    #         "query": sent,
    #         "title": "2016年美国总统大选期间，受访选民平均每人每天接触到4篇虚假新闻，虚假新闻被认为影响了2016年美国大选和英国脱欧的投票结果",
    #         "label": 0,
    #     },
    #     {"query": "今天上班", "title": "今天不上班", "label": 0},
    #     {"query": "今天上班", "title": "今天还是要上班", "label": 0},
    # ]

def generate_data(sent, data_list):
    return [[data[2], sent] for data in data_list]
    # return  [
        #     ['疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例', '法国火鸡养殖场发现高致病性禽流感病例。'], 
        #     ['2016年美国总统大选期间，受访选民平均每人每天接触到4篇虚假新闻，虚假新闻被认为影响了2016年美国大选和英国脱欧的投票结果；近期，在新型冠状病毒感染的肺炎疫情防控的关键期，在全国人民都为疫情揪心时，网上各种有关疫情防控的谣言接连不断，从“广州公交线路因新型冠状病毒肺炎疫情停运”到“北京市为防控疫情采取封城措施”，从“钟南山院士被感染”到“10万人感染肺炎”等等，这些不切实际的谣言，“操纵”了舆论感情，误导了公众的判断，更影响了社会稳定。', '平均每人每天接触到4篇虚假新闻'],
        #     ['一群女性跑马拉松。', '他们中的一个将赢得比赛']
        # ]
    
