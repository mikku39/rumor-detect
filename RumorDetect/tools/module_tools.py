import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Linear, Embedding
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F


ernie_bot_token = ""


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


# CNN 模型
class SimpleConvPool(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,  # 通道数
        num_filters,  # 卷积核数量
        filter_size,  # 卷积核大小
        batch_size=None,
    ):  # 16
        super(SimpleConvPool, self).__init__()
        self.batch_size = batch_size
        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            act="tanh",
        )

        self._pool2d = fluid.dygraph.Pool2D(
            pool_size=(150 - filter_size[0] + 1, 1),  # 池化核大小，这里需要计算
            pool_type="max",  # 池化类型，默认是最大池化,padding = [1,1],
            pool_stride=1,
        )  # 池化步长

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        x = fluid.layers.reshape(x, shape=[self.batch_size, -1])
        return x

class CNN(fluid.dygraph.Layer):
    def __init__(self, batch_size=1):
        super(CNN, self).__init__()
        self.dict_dim = 4409
        self.emb_dim = 128  # emb纬度
        self.hid_dim = [32]  # 卷积核数量
        self.fc_hid_dim = 96  # fc参数纬度
        self.class_dim = 2  # 分类数
        self.channels = 1  # 输入通道数
        self.win_size = [[3, 128]]  # 卷积核尺寸
        self.batch_size = batch_size
        self.seq_len = 150
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim], dtype="float32", is_sparse=False
        )
        self._simple_conv_pool_1 = SimpleConvPool(
            self.channels,  # 通道数
            self.hid_dim[0],  # 卷积核数量
            self.win_size[0],  # 卷积核尺寸
            batch_size=self.batch_size,  # 批次大小
        )
        self._fc1 = Linear(
            input_dim=self.hid_dim[0], output_dim=self.fc_hid_dim, act="tanh"
        )
        self._fc_prediction = Linear(
            input_dim=self.fc_hid_dim, output_dim=self.class_dim, act="softmax"
        )

    def forward(self, inputs, label=None):

        emb = self.embedding(inputs)  # [2400, 128]
        emb = fluid.layers.reshape(  # [16, 1, 150, 128]
            emb, shape=[-1, self.channels, self.seq_len, self.emb_dim]
        )
        conv_3 = self._simple_conv_pool_1(emb)
        fc_1 = self._fc1(conv_3)
        prediction = self._fc_prediction(fc_1)
        if label is not None:
            acc = fluid.layers.accuracy(prediction, label=label)
            return prediction, acc
        else:
            return prediction


# 语义匹配模型
class PointwiseMatching(nn.Layer):
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(
        self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None
    ):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(
            input_ids, token_type_ids, position_ids, attention_mask
        )

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        probs = F.softmax(logits)  # qaqaqqaq

        return probs
