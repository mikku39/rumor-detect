from RumorDetect import RumorDetect
instance = RumorDetect.rumor_detect(
    news_mode=["bing"],
    summary_mode=["baidu"],
    compare_mode=["baidu", "match"],
    judge_mode=["cnn"],
)
# sent = "华晨宇有小孩子了"
# sent = "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例"
sent = "狂飙兄弟在连云港遇到鬼秤"
instance.run(sent)