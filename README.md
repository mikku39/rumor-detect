kk的ysu毕设项目。微博谣言检测
## 安装教程
```shell
git clone https://github.com/mikku39/rumor-detect.git
cd rumor-detect
sudo pip install .
```


## 快速启动
```python
from RumorDetect.RumorDetect import rumor_detect

instance = rumor_detect()
sent = "这是谣言吗"
instance.run(sent)
```

## 前端页面
```shell
RumorDetect serve
```