# Welcome to Rumor Detect Docs

这里是ykk的ysu毕业设计项目——微博谣言检测的文档。
将会包含API文档和使用文档

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