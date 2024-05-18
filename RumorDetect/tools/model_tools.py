import numpy as np
import paddle
import requests

from RumorDetect.tools.data_tools import check_and_download, get_default_path
from paddlenlp.transformers import AutoModelForConditionalGeneration
import paddle
import paddlenlp
import paddle.fluid as fluid
from RumorDetect.model import CNN, PointwiseMatching

from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
import paddlehub as hub
import os

# 文本的最大长度
max_source_length = 128
# 摘要的最大长度
max_target_length = 64
# 摘要的最小长度
min_target_length = 0

ernie_bot_token = ""

# 针对单条文本，生成摘要
def pegasus_single_infer(text, summary_model, summary_tokenizer):
    tokenized = summary_tokenizer(
        text, truncation=True, max_length=max_source_length, return_tensors="pd"
    )
    preds, _ = summary_model.generate(
        input_ids=tokenized["input_ids"],
        max_length=max_target_length,
        min_length=min_target_length,
        decode_strategy="beam_search",
        num_beams=4,
    )
    summary_ans = summary_tokenizer.decode(
        preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return summary_ans


# 返回语义相似度
def match_infer(model, data_loader):

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


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids

def pegasus_init():
    path = f"{get_default_path()}/pegasus_checkpoints"
    url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/pegasus_checkpoints.zip"
    check_and_download(path, url)
    summary_model = AutoModelForConditionalGeneration.from_pretrained(path)
    return summary_model

def match_init():
    # 语义匹配网络
    pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained(
        "ernie-gram-zh"
    )
    compare_model = PointwiseMatching(pretrained_model)
    model_path = f"{get_default_path()}/point_wise.pdparams"
    url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/point_wise.pdparams"
    check_and_download(model_path, url)
    state_dict = paddle.load(model_path)
    compare_model.set_dict(state_dict)
    return compare_model

def entailment_init():
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
    paddle.enable_static()
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

    compare_model = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=["acc"],
    )
    compare_model.max_train_steps = 9999
    paddle.disable_static()
    return compare_model

def cnn_init():
    # cnn模型
    judge_model = CNN(batch_size=1)
    model_path = f"{get_default_path()}/save_dir_9240.pdparams"
    model_url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/save_dir_9240.pdparams"
    check_and_download(model_path, model_url)
    cnn_model, _ = fluid.load_dygraph(model_path)
    judge_model.load_dict(cnn_model)
    judge_model.eval()
    dict_path = f"{get_default_path()}/dict.txt"
    dict_url = (
        "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/dict.txt"
    )
    check_and_download(dict_path, dict_url)
    return judge_model

def ernie_bot_init():
    global ernie_bot_token
    if ernie_bot_token == "":
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": os.environ.get("BAIDU_MODEL_KEY"), "client_secret": os.environ.get("BAIDU_MODEL_SECRET")}
        ernie_bot_token = str(requests.post(url, params=params).json().get("access_token"))
    return ernie_bot_token