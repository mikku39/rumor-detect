from RumorDetect.RumorDetect import rumor_detect

instance = rumor_detect(
    news_mode=["bing"],
    summary_mode=["ernie_bot"],
    compare_mode=["ernie_bot", "match"],
    judge_mode=["cnn", "ernie_bot"],
)
sent = "狂飙兄弟没有在连云港遇到鬼秤"
instance.run(sent)