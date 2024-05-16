from tools.data_tools import (
    data2np,
    tx_search,
    get_news_list,
    generate_data_source,
    generate_data,
    google_search,
)
import jieba.analyse
from paddlenlp.datasets import MapDataset
import paddle
from tools.model_tools import convert_example
import paddlenlp
from tools.model_tools import match_infer, pegasus_single_infer
import numpy as np
from paddlenlp.data import Pad, Tuple
from functools import partial


# 返回新闻 list
def tianxing_find_news(keyword_list, keyword_limit_num=5, news_limit_num=1):
    if len(keyword_list) > keyword_limit_num:
        keyword_list = keyword_list[:keyword_limit_num]
    tx_data = tx_search(keyword_list)
    if tx_data['code'] != 200:
        return []
    tx_data = tx_data["result"]["newslist"]
    if len(tx_data) > news_limit_num:
        tx_data = tx_data[:news_limit_num]
    return get_news_list(tx_data)


def google_find_news(keyword_list, keyword_limit_num=5, news_limit_num=1):
    if len(keyword_list) > keyword_limit_num:
        keyword_list = keyword_list[:keyword_limit_num]
    keyword_str = " ".join(keyword_list)
    google_data = google_search(keyword_str)
    for data in google_data:
        data["url"] = data["formattedUrl"]
    if len(google_data) > news_limit_num:
        google_data = google_data[:news_limit_num]
    return get_news_list(google_data)


# 查找关键词
def get_keywords(sent):
    """
    通过 jieba.analyse.extract_tags 查找微博文本输出关键词列表。

    Args:
        row: 一个 str。
    """
    res = jieba.analyse.extract_tags(sent)
    return res


def pegasus_group_infer(summary_model, summary_tokenizer, sent, news_list):
    summary_list = []
    for news in news_list:
        text = news[2]
        summary_text = pegasus_single_infer(text, summary_model, summary_tokenizer)
        summary_list.append((news[0], news[1], summary_text))
    sent = pegasus_single_infer(sent, summary_model, summary_tokenizer)
    return sent, summary_list


def check_match(match_model, sent, news_list):
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
    for idx, y_pred in enumerate(y_preds):
        text_pair = test_ds[idx]
        text_pair["label"] = y_pred
        print(text_pair)


def check_entailment(entailment_model, sent, news_list):
    inv_label_map = {0: "谣言", 1: "非谣言", 2: "不相关"}
    paddle.enable_static()
    data_source = generate_data(sent, news_list)
    run_states = entailment_model.predict(data=data_source)
    results = [run_state.run_results for run_state in run_states]
    index = 0
    for batch_result in results[0][0]:
        print(batch_result)
        batch_result = np.argmax(batch_result)
        print(
            f"source:{data_source[index][1]}, \
                news:{data_source[index][0]}  \
                predict={inv_label_map[batch_result]}  \
                news_url:{news_list[index][1]}"
        )
        index += 1

    paddle.disable_static()


def cnn_infer(model, sent):
    lab = ["谣言", "非谣言"]
    infer_np_doc = data2np(sent)
    result = model(infer_np_doc)
    print("CNN 模型预测结果为：", lab[np.argmax(result.numpy())])
