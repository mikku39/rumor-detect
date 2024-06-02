# Tutorial

## 可能需要配置的环境变量（各种API的token

### 新闻搜索  
- `TJSX_API_KEY`  
- `CSE_ID` 和 `CSE_API_KEY`  
- `BING_SEARCH_KEY`  
- `URL2IO_KEY`  

### summary and compare
-  `BAIDU_API_KEY` 和 `BAIDU_API_SECRET`  
-  `ERNIE_BOT_KEY` 和 `ERNIE_BOT_SECRET`  

### judge
-  `ERNIE_BOT_KEY` 和 `ERNIE_BOT_SECRET`  


## Example

```python
from RumorDetect.RumorDetect import rumor_detect

instance = rumor_detect(
    news_mode=["google"],
    summary_mode=["ernie_bot"],
    compare_mode=["ernie_bot", "match"],
    judge_mode=["cnn", "ernie_bot"],
)
sent = "这是谣言吗"
instance.run(sent)
```

## Debug Example
```python
from RumorDetect.RumorDetect import rumor_detect

instance = rumor_detect(
    news_mode=["google"],
    summary_mode=["ernie_bot"],
    compare_mode=["ernie_bot", "match"],
    judge_mode=["cnn", "ernie_bot"],
)
sent = "这是谣言吗"
tmp = instance.debug_run(sent)
next(tmp)
print(instance.get_intermediate())
instance.update_params({"sent": "这是谣言吗", "keywords": ["这是谣言吗"]})
print(instance.get_intermediate())
instance.set
next(tmp)
print(instance.get_intermediate())
next(tmp)
print(instance.get_intermediate())
next(tmp)
```


## 修改自己模块的 Example
```python
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

```