from functools import partial
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import paddle
import requests
from RumorDetect.model import BaseCompareModel
import paddlenlp

from RumorDetect.tools.data_tools import (
    generate_data,
    generate_data_source,
    truncate_bytes,
)
from RumorDetect.component import check_and_download, get_default_path, get_env
from RumorDetect.tools.module_tools import PointwiseMatching, convert_example
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from RumorDetect.tools import module_tools


class MatchingCompareModel(BaseCompareModel):
    '''
        使用 Ernie-gram 模型配合 Pointwise匹配进行文本比较
        比较两个文本的语义是否相同。
        由于谣言检测更多的情况下是对比微博文本是否是新闻文本的子集。
        因此概述下来，新闻文本包含微博文本的情况下，两者语义不一定能够相似
        所以准确率可能不高。
    '''
    def __init__(self) -> None:
        self.init()

    def init(self):
        pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained(
            "ernie-gram-zh"
        )
        self.compare_model = PointwiseMatching(pretrained_model)
        model_path = f"{get_default_path()}/point_wise.pdparams"
        url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/point_wise.pdparams"
        check_and_download(model_path, url)
        state_dict = paddle.load(model_path)
        self.compare_model.set_dict(state_dict)

    def compare(self, sent: str, news_list: List[Tuple]) -> List[Dict]:
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
        y_probs = self.match_infer(self.compare_model, predict_data_loader)
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

    def match_infer(self, model, data_loader):

        batch_probs = []

        # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
        model.eval()

        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
                batch_prob = model(
                    input_ids=input_ids, token_type_ids=token_type_ids
                ).numpy()

                batch_probs.append(batch_prob)
            batch_probs = np.concatenate(batch_probs, axis=0)

            return batch_probs


class EntailmentCompareModel(BaseCompareModel):
    '''
        使用 Bert 模型进行文本比较
        这里主要是训练了文本的蕴含关系。因此我们比较新闻文本是否蕴含了微博文本的信息。
        准确率中等。
    '''
    def __init__(self) -> None:
        self.init()

    def init(self):
        import paddlehub as hub
        paddle.enable_static()
        module = hub.Module("bert_chinese_L-12_H-768_A-12")

        path = f"{get_default_path()}/hub_finetune_ckpt"
        url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/hub_finetune_ckpt.zip"
        check_and_download(path, url)

        class MyNews(BaseNLPDataset):
            def __init__(self):
                # 数据集存放位置
                dict_path = f"{get_default_path()}/dict.txt"
                dict_url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/dict.txt"
                check_and_download(dict_path, dict_url)
                super(MyNews, self).__init__(
                    base_path=get_default_path(),
                    train_file="dict.txt",
                    dev_file="dict.txt",
                    test_file="dict.txt",
                    train_file_with_header=True,
                    dev_file_with_header=True,
                    test_file_with_header=True,
                    label_list=["contradiction", "entailment", "neutral"],
                )

        dataset = MyNews()
        reader = hub.reader.ClassifyReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            sp_model_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path(),
            max_seq_len=128,
        )

        strategy = hub.AdamWeightDecayStrategy(
            weight_decay=0.001,
            warmup_proportion=0.1,
            learning_rate=5e-5,
        )

        config = hub.RunConfig(
            use_cuda=True,
            num_epoch=2,
            batch_size=32,
            checkpoint_dir=path,
            strategy=strategy,
        )
        inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
        # 配置fine-tune任务
        # Use "pooled_output" for classification tasks on an entire sentence.
        pooled_output = outputs["pooled_output"]

        feed_list = [
            inputs["input_ids"].name,
            inputs["position_ids"].name,
            inputs["segment_ids"].name,
            inputs["input_mask"].name,
        ]

        self.compare_model = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config,
            metrics_choices=["acc"],
        )
        self.compare_model.max_train_steps = 9999
        paddle.disable_static()

    def compare(self, sent: str, news_list: List[Tuple]) -> List[Dict]:
        inv_label_map = {0: "谣言", 1: "非谣言", 2: "不相关"}
        paddle.enable_static()
        data_source = generate_data(sent, news_list)
        run_states = self.compare_model.predict(data=data_source)
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


class BaiduCompareModel(BaseCompareModel):
    '''
        使用百度短文本比较接口进行文本比较
        由于谣言检测更多的情况下是对比微博文本是否是新闻文本的子集。
        因此概述下来，新闻文本包含微博文本的情况下，两者语义不一定能够相似
        所以准确率可能不高。
        需要配置环境变量：BAIDU_API_KEY 和 BAIDU_API_SECRET
    '''
    def __init__(self) -> None:
        self.init()

    def init(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": get_env("BAIDU_API_KEY"),
            "client_secret": get_env("BAIDU_API_SECRET"),
        }
        self.token = str(requests.post(url, params=params).json().get("access_token"))

    def compare(self, sent: str, news_list: List[Tuple]) -> List[Dict]:
        inv_label_map = {0: "谣言", 1: "非谣言"}
        url = (
            "https://aip.baidubce.com/rpc/2.0/nlp/v2/simnet?charset=UTF-8&access_token="
            + self.token
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


class ErnieBotCompareModel(BaseCompareModel):
    '''
        使用 ErnieBot 模型进行文本比较
        配置的是文本蕴含的prompt。因此我们比较新闻文本是否蕴含了微博文本的信息。
        需要配置环境变量：ERNIE_BOT_KEY 和 ERNIE_BOT_SECRET
    '''
    def __init__(self) -> None:
        self.init()

    def init(self):
        if module_tools.ernie_bot_token == "":
            url = "https://aip.baidubce.com/oauth/2.0/token"
            params = {
                "grant_type": "client_credentials",
                "client_id": get_env("ERNIE_BOT_KEY"),
                "client_secret": get_env("ERNIE_BOT_SECRET"),
            }
            self.token = str(
                requests.post(url, params=params).json().get("access_token")
            )
            module_tools.ernie_bot_token = self.token
        else:
            self.token = module_tools.ernie_bot_token

    def compare(self, sent: str, news_list: List[Tuple]) -> List[Dict]:
        inv_label_map = {0: "谣言", 1: "非谣言"}
        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
            + self.token
        )
        result_list = []
        for news in news_list:
            payload = json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"文本A:{news[2]}\n文本B: {sent}",
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
