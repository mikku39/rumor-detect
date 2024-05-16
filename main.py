from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers import AutoModelForConditionalGeneration
import paddle
import paddlenlp
import paddle.fluid as fluid
from model import CNN, PointwiseMatching
from dataclasses import dataclass
from typing import Callable

from component import (
    get_keywords,
    pegasus_group_infer,
    check_match,
    cnn_infer,
    check_entailment,
    tianxing_find_news,
    google_find_news,
)
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from dotenv import load_dotenv
from tools.data_tools import get_default_path, check_and_download


@dataclass
class model_template:
    init: Callable
    infer: Callable


class rumor_detect:
    def __init__(
        self,
        auto_init=True,
        enable_summary=True,
        news_mode="google",
        summary_mode="pegasus",
        compare_mode="entailment",
        judge_mode="cnn",
        keyword_limit_num=8,
        news_limit_num=5,
    ):
        load_dotenv()
        self.enable_summary = enable_summary
        self.news_mode = news_mode
        self.summary_mode = summary_mode
        self.compare_mode = compare_mode
        self.judge_mode = judge_mode
        self.keyword_limit_num = keyword_limit_num
        self.news_limit_num = news_limit_num
        self.auto_init = auto_init
        self.initialized = False

        self.find_news_dict = {  # 多种新闻搜索方式
            "tianxing": tianxing_find_news,  # 天聚数行API
            "google": google_find_news,  # google搜索
        }

        self.summary_dict = {
            "pegasus": model_template(
                init=self.pegasus_init, infer=pegasus_group_infer
            ),
        }

        self.compare_dict = {  # 多种新闻比较方式
            "entailment": model_template(
                init=self.entailment_init, infer=check_entailment
            ),  # 语义蕴含
            "match": model_template(
                init=self.match_init, infer=check_match
            ),  # 语义匹配
        }

        self.judge_dict = {  # 多种模型判断方式
            "cnn": model_template(init=self.cnn_init, infer=cnn_infer),
        }

        if self.auto_init:
            self.init()
            self.initialized = True

    def init(self):
        if self.enable_summary:
            self.summary_dict[self.summary_mode].init()
        self.compare_dict[self.compare_mode].init()
        self.judge_dict[self.judge_mode].init()

    def run(self, sent):
        if not self.initialized:
            print("未初始化,正在按各模块的 mode 配置初始化")
            self.init()
        keywords = get_keywords(sent)
        news_list = self.find_news_dict[self.news_mode](
            keywords, self.keyword_limit_num, self.news_limit_num
        )
        if not news_list:
            print("未找到相关新闻")
        else:
            if self.enable_summary:
                sent, news_list = self.summary_dict[self.summary_mode].infer(
                    self.summary_model, self.summary_tokenizer, sent, news_list
                )
            print(news_list)
            self.compare_dict[self.compare_mode].infer(
                self.compare_model, sent, news_list
            )
        print(keywords)
        self.judge_dict[self.judge_mode].infer(self.judge_model, sent)

    def pegasus_init(self):
        path = f"{get_default_path()}/pegasus_checkpoints"
        url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/pegasus_checkpoints.zip"
        check_and_download(path, url)
        self.summary_model = AutoModelForConditionalGeneration.from_pretrained(path)
        self.summary_tokenizer = AutoTokenizer.from_pretrained(path)

    def match_init(self):
        # 语义匹配网络
        pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained(
            "ernie-gram-zh"
        )
        self.compare_model = PointwiseMatching(pretrained_model)
        model_path = f"{get_default_path()}/point_wise.pdparams"
        url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/point_wise.pdparams"
        check_and_download(model_path, url)
        state_dict = paddle.load(model_path)
        self.compare_model.set_dict(state_dict)

    def entailment_init(self):
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

    def cnn_init(self):
        # cnn模型
        self.judge_model = CNN(batch_size=1)
        model_path = f"{get_default_path()}/save_dir_9240.pdparams"
        model_url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/save_dir_9240.pdparams"
        check_and_download(model_path, model_url)
        cnn_model, _ = fluid.load_dygraph(model_path)
        self.judge_model.load_dict(cnn_model)
        self.judge_model.eval()
        dict_path = f"{get_default_path()}/dict.txt"
        dict_url = (
            "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/dict.txt"
        )
        check_and_download(dict_path, dict_url)

    def list_summary_mode(self):
        return self.summary_dict.keys()

    def list_compare_mode(self):
        return self.compare_dict.keys()

    def list_judge_mode(self):
        return self.judge_dict.keys()

    def list_news_mode(self):
        return self.find_news_dict.keys()

    def add_summary_mode(self, key, value):
        self.summary_dict[key] = value

    def add_compare_mode(self, key, value):
        self.compare_dict[key] = value

    def add_judge_mode(self, key, value):
        self.judge_dict[key] = value

    def add_news_mode(self, key, value):
        self.find_news_dict[key] = value

    def set_summary_mode(self, mode):
        self.summary_mode = mode

    def set_compare_mode(self, mode):
        self.compare_mode = mode

    def set_judge_mode(self, mode):
        self.judge_mode = mode

    def set_news_mode(self, mode):
        self.news_mode = mode


if __name__ == "__main__":
    # instance = rumor_detect(enable_summary= False)
    # instance = rumor_detect(news_mode="tianxing")
    instance = rumor_detect(news_mode="google", compare_mode="match")
    # sent = "华晨宇有小孩子了"
    sent = "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例"
    instance.run(sent)
