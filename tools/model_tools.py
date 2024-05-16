import numpy as np
import paddle

# 文本的最大长度
max_source_length = 128
# 摘要的最大长度
max_target_length = 64
# 摘要的最小长度
min_target_length = 0


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

