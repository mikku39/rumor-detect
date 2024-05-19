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