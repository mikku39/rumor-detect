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
)
import jieba.analyse
from paddlenlp.datasets import MapDataset
import paddle
from RumorDetect.tools.model_tools import convert_example, match_infer, pegasus_single_infer
import paddlenlp
import numpy as np
from paddlenlp.data import Pad, Tuple
from functools import partial
from paddlenlp.transformers import AutoTokenizer
import os
import json

# 返回新闻 list
def tianxing_find_news(keyword_list, keyword_limit_num=8, news_limit_num=5, banned_url=["zhihu", "baijiahao"]):
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


def google_find_news(keyword_list, keyword_limit_num=8, news_limit_num=5, banned_url=["zhihu", "baijiahao"]):
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

def bing_find_news(keyword_list, keyword_limit_num=8, news_limit_num=5, banned_url=[]):
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

def data_ban_url(data, banned_url, news_limit_num):
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
    sent = pegasus_single_infer(sent, summary_model, summary_tokenizer)
    return sent, summary_list

def baidu_summary_group_infer(summary_model, sent, news_list):
    summary_list = []
    for news in news_list:
        text = news[2]
        summary_text = baidu_summary_single_infer(text)
        summary_list.append((news[0], news[1], summary_text))
    sent = baidu_summary_single_infer(sent)
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
                "source": test_ds[idx]['query'],
                "news": test_ds[idx]['title'],
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

def baidu_compare(model, sent, news_list):
    inv_label_map = {0: "谣言", 1: "非谣言"}
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": os.environ.get("BAIDU_KEY"), "client_secret": os.environ.get("BAIDU_SECRET")}
    token = str(requests.post(url, params=params).json().get("access_token"))
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v2/simnet?charset=UTF-8&access_token=" + token
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    result_list = []
    for news in news_list:
        payload = json.dumps({
            "text_1": truncate_bytes(sent),
            "text_2": truncate_bytes(news[2]),
        })
        response = requests.request("POST", url, headers=headers, data=payload)
        result = response.json()
        if "error_code" in result:
            print(f"使用百度短文本比较的时候出错：{result}")
            return result_list
        result_list.append({
            "source": sent,
            "news": news[2],
            "news_url": news[1],
            "score": result['score'],
            "predict": inv_label_map[result['score'] > 0.6],
        })
    # print(tabulate.tabulate(result_list, headers="keys", tablefmt="grid"))
    return result_list
    

def cnn_infer(model, sent):
    lab = ["谣言", "非谣言"]
    infer_np_doc = data2np(sent)
    result = model(infer_np_doc)
    print("CNN 模型预测结果为：", lab[np.argmax(result.numpy())])
    return {"source": sent, "score": result.numpy()[0][1], "predict": lab[np.argmax(result.numpy())]}
