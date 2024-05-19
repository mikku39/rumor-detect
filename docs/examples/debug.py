from RumorDetect.RumorDetect import rumor_detect
instance = rumor_detect(
    news_mode=["google"],
    summary_mode=["ernie_bot"],
    compare_mode=["ernie_bot", "match"],
    judge_mode=["cnn", "ernie_bot"],
)
# sent = "华晨宇有小孩子了"
# sent = "疫情在欧洲蔓延！法国一养殖场发现高致病性禽流感病例"
sent = "狂飙兄弟在连云港遇到鬼秤"
tmp = instance.debug_run(sent)
print(tmp)