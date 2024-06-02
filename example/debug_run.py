from RumorDetect.RumorDetect import rumor_detect

instance = rumor_detect(
    news_mode=["bing"],
    summary_mode=["ernie_bot"],
    compare_mode=["entailment", "ernie_bot"],
    judge_mode=["cnn", "ernie_bot"],
)
# sent = "朝鲜使用气球向韩国投放垃圾"
sent = "隔夜西瓜不能吃，会中毒"
tmp = instance.debug_run(sent)
next(tmp)
print(instance.get_intermediate())
instance.update_params({"sent": "这是谣言吗", "keywords": ["这是谣言吗"]})
print(instance.get_intermediate())
next(tmp)
print(instance.get_intermediate())
next(tmp)
print(instance.get_intermediate())
next(tmp)