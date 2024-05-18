import requests
from RumorDetect.tools.data_tools import (
    bing_search,
    data2np,
    get_default_path,
    truncate_bytes,
    tx_search,
    get_news_list,
    generate_data_source,
    generate_data,
    google_search,
    baidu_summary_single_infer,
    ernie_bot_summary_single_infer,
    bing_spider_search,
)
import jieba.analyse
from paddlenlp.datasets import MapDataset
import paddle
from RumorDetect.tools.model_tools import (
    convert_example,
    match_infer,
    pegasus_single_infer,
)
import paddlenlp
import numpy as np
from paddlenlp.data import Pad, Tuple
from functools import partial
from paddlenlp.transformers import AutoTokenizer
import os
import json
from typing import List, Dict



def tianxing_find_news(
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


def google_find_news(
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


def bing_find_news(
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


def bing_spider_find_news(
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


def data_ban_url(
    data: List[Dict[str, str]], banned_url: List[str], news_limit_num: int
):
    '''
        根据 banned_url 列表过滤 data 中的新闻
    '''
    if len(data) < news_limit_num:
        return data
    for banned in banned_url:
        for new in data:
            if banned in new["url"]:
                data.remove(new)
            if len(data) < news_limit_num:
                return data
    return data


# 查找关键词
def get_keywords(sent):
    """
    通过 jieba.analyse.extract_tags 查找微博文本输出关键词列表。

    Args:
        row: 一个 str。
    """
    res = jieba.analyse.extract_tags(sent)
    return res


def pegasus_group_infer(summary_model, sent, news_list):
    summary_list = []
    path = f"{get_default_path()}/pegasus_checkpoints"
    summary_tokenizer = AutoTokenizer.from_pretrained(path)
    for news in news_list:
        text = news[2]
        summary_text = pegasus_single_infer(text, summary_model, summary_tokenizer)
        summary_list.append((news[0], news[1], summary_text))

    if len(sent) > 20:
        sent = pegasus_single_infer(sent, summary_model, summary_tokenizer)
    return sent, summary_list


def baidu_summary_group_infer(summary_model, sent, news_list):
    summary_list = []
    for news in news_list:
        text = news[2]
        summary_text = baidu_summary_single_infer(text)
        summary_list.append((news[0], news[1], summary_text))

    if len(sent) > 20:
        sent = baidu_summary_single_infer(sent)

    return sent, summary_list


def ernie_bot_summary_group_infer(token, sent, news_list):
    summary_list = []
    for news in news_list:
        text = news[2]
        summary_text = ernie_bot_summary_single_infer(token, text)
        summary_list.append((news[0], news[1], summary_text))
    if len(sent) > 20:
        sent = ernie_bot_summary_single_infer(token, sent)
    return sent, summary_list


def check_match(match_model, sent, news_list):
    inv_label_map = {0: "谣言", 1: "非谣言"}
    match_tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained(
        "ernie-gram-zh"
    )
    # 预测数据的转换函数
    trans_func = partial(
        convert_example,
        tokenizer=match_tokenizer,
        max_seq_length=512,
        is_test=True,
    )

    # 预测数据的组 batch 操作
    # predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=match_tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=match_tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    data_source = generate_data_source(sent, news_list)
    test_ds = MapDataset(data_source)
    batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)
    # 生成预测数据 data_loader
    predict_data_loader = paddle.io.DataLoader(
        dataset=test_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
    )
    # 执行预测函数
    y_probs = match_infer(match_model, predict_data_loader)
    y_preds = np.argmax(y_probs, axis=1)
    test_ds = MapDataset(data_source)
    result_list = []
    for idx, y_pred in enumerate(y_preds):
        result_list.append(
            {
                "source": test_ds[idx]["query"],
                "news": test_ds[idx]["title"],
                "news_url": news_list[idx][1],
                "score": y_probs[idx][1],
                "predict": inv_label_map[y_pred],
            }
        )
    # print(tabulate.tabulate(result_list, headers="keys", tablefmt="grid"))
    return result_list


def check_entailment(entailment_model, sent, news_list):
    inv_label_map = {0: "谣言", 1: "非谣言", 2: "不相关"}
    paddle.enable_static()
    data_source = generate_data(sent, news_list)
    run_states = entailment_model.predict(data=data_source)
    results = [run_state.run_results for run_state in run_states]
    index = 0
    result_list = []
    for batch_result in results[0][0]:
        print(batch_result)
        score = batch_result[1]
        batch_result = np.argmax(batch_result)
        result_list.append(
            {
                "source": data_source[index][1],
                "news": data_source[index][0],
                "news_url": news_list[index][1],
                "score": score,
                "predict": inv_label_map[batch_result],
            }
        )
        index += 1
    # print(tabulate.tabulate(result_list, headers="keys", tablefmt="grid"))
    paddle.disable_static()
    return result_list


def baidu_match(model, sent, news_list):
    inv_label_map = {0: "谣言", 1: "非谣言"}
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": os.environ.get("BAIDU_API_KEY"),
        "client_secret": os.environ.get("BAIDU_API_SECRET"),
    }
    token = str(requests.post(url, params=params).json().get("access_token"))
    url = (
        "https://aip.baidubce.com/rpc/2.0/nlp/v2/simnet?charset=UTF-8&access_token="
        + token
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    result_list = []
    for news in news_list:
        payload = json.dumps(
            {
                "text_1": truncate_bytes(sent),
                "text_2": truncate_bytes(news[2]),
            }
        )
        response = requests.request("POST", url, headers=headers, data=payload)
        result = response.json()
        if "error_code" in result:
            print(f"使用百度短文本比较的时候出错：{result}")
            return result_list
        result_list.append(
            {
                "source": sent,
                "news": news[2],
                "news_url": news[1],
                "score": result["score"],
                "predict": inv_label_map[result["score"] > 0.6],
            }
        )
    # print(tabulate.tabulate(result_list, headers="keys", tablefmt="grid"))
    return result_list


def ernie_bot_entailment(token, sent, news_list):
    inv_label_map = {0: "谣言", 1: "非谣言"}
    url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
        + token
    )
    result_list = []
    for news in news_list:
        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "文本A:法国一养殖场发现高致病性禽流感病例\n文本B: 法国火鸡养殖场发现高致病性禽流感病例",
                    }
                ],
                "temperature": 0.95,
                "top_p": 0.8,
                "penalty_score": 1,
                "system": "【功能说明】 你是一个高级人工智能助手，专门设计来评估两段文本的关联性和真实性。你的任务是在假定第一段文本（文本A）为真的情况下，分析并给出第二段文本（文本B）为真的置信度。置信度是一个介于0到1之间的数值，高置信度（接近1）表示文本B很可能也是真实的，而低置信度（接近0）则表示文本B很可能是虚假的。  【输入格式要求】  输入必须包含两个独立的文本段落，分别标记为“文本A”和“文本B”。 每段文本应该简洁明了，长度不超过500字。 文本应直接输入，不含任何格式化元素（如HTML标签）或非文本内容。 【操作要求】  验证输入格式符合上述要求。 在确认文本A为真的基础上，使用数据和逻辑分析文本B的真实性。 输出一个形如“【0.xxx】”的置信度数值，表示文本B的真实性。 确保输出仅包含置信度数值，不添加任何解释或其他文本。 【示例输入】 文本A: 乔治·华盛顿是美国的第一任总统。 文本B: 托马斯·杰斐逊是美国的第三任总统。  【示例输出】 【0.999】",
                "stop": ["】"],
                "disable_search": True,
                "enable_citation": False,
                "response_format": "text",
            }
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        result = response.json()
        try:
            score = float(json.loads(response.text)["result"].strip("【】"))
        except Exception as e:
            print(f"baidu_model_predict_ERROR: {response.text}")
            return result_list
        result_list.append(
            {
                "source": sent,
                "news": news[2],
                "news_url": news[1],
                "score": score,
                "predict": inv_label_map[score > 0.6],
            }
        )
    return result_list


def cnn_infer(model, sent):
    lab = ["谣言", "非谣言"]
    infer_np_doc = data2np(sent)
    result = model(infer_np_doc)
    result_dict = {
        "source": sent,
        "score": result.numpy()[0][1],
        "predict": lab[np.argmax(result.numpy())],
    }
    print("CNN 模型预测结果为：", result_dict)
    return result_dict


def ernie_bot_infer(token, sent):
    lab = ["谣言", "非谣言"]

    url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
        + token
    )

    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": sent,
                }
            ],
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1,
            "system": "【功能说明】 你是一个专门设计来验证信息真实性的人工智能助手，通过结合百度搜索引擎获取的信息来评估输入信息的真实性。你的任务是给出一个介于0到1之间的置信度数值，这个数值反映了输入信息非谣言的概率。高置信度（接近1）表示信息很可能是真实的，低置信度（接近0）表示信息很可能是虚假的。  【操作要求】  用户提供一条信息。 你通过搜索引擎进行搜索，分析搜索结果。 仅返回一个形如“【0.xxx】”的置信度数值，该数值基于搜索结果的相关性和信源的权威性。 不要添加任何解释或其他文本。",
            "disable_search": False,
            "enable_citation": False,
            "response_format": "json",
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        score = float(json.loads(response.text)["result"].strip("【】"))
    except Exception as e:
        print(f"baidu_model_predict_ERROR: {response.text}")
        return {"source": sent, "score": 0.5, "predict": "谣言"}
    result_dict = {"source": sent, "score": score, "predict": lab[score > 0.6]}
    print(f"baidu_model_predict: {result_dict}")
    return result_dict
