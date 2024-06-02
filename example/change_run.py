from RumorDetect.RumorDetect import rumor_detect

from RumorDetect.model import BaseSummaryModel


class test(BaseSummaryModel):
    def init(self):
        pass

    def get_summary(self, sent: str, news_list: list) -> tuple:
        return "没错", [("错了", "http://www.baidu.com", "错了")]


instance = rumor_detect(
    news_mode=["bing"],
    compare_mode=["entailment", "ernie_bot"],
    judge_mode=["cnn", "ernie_bot"],
    auto_init=False,
)
instance.add_summary_mode("test", test)
instance.set_summary_mode(["test"])
sent = "隔夜西瓜不能吃，会中毒"
instance.run(sent)
