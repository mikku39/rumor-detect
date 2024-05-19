## 主模块
---
负责调度其他模块执行  
现有的模块取值：

- news_mode
    - **tjsx**：天聚数行的 API 搜索，需配置环境变量`TJSX_API_KEY`
    - **google**：google 的 API 搜索，需配置环境变量`CSE_ID` 和 `CSE_API_KEY`
    - **bing**：bing 的 API 搜索，需配置环境变量`BING_SEARCH_KEY`
    - **bing_spider**：bing 的爬虫搜索
- summary_mode
    - **pegasus**：Pegasus 天马模型概要
    - **baidu**：Baidu Api 概要，需配置环境变量`BAIDU_API_KEY` 和 `BAIDU_API_SECRET`
    - **ernie_bot**：ErnieBot 大模型概要，需配置环境变量`ERNIE_BOT_KEY` 和 `ERNIE_BOT_SECRET`
- compare_mode
    - **entailment**：语义蕴含判断
    - **match**：语义匹配判断
    - **baidu**：Baidu Api 语义匹配判断，需配置环境变量`BAIDU_API_KEY` 和 `BAIDU_API_SECRET`
    - **ernie_bot**：ErnieBot 大模型语义蕴含判断，需配置环境变量`ERNIE_BOT_KEY` 和 `ERNIE_BOT_SECRET`
-  judge_mode
    - **cnn**：CNN 模型判断
    - **ernie_bot**：ErnieBot 大模型判断，需配置环境变量`ERNIE_BOT_KEY` 和 `ERNIE_BOT_SECRET`

# rumor_detect

::: RumorDetect.RumorDetect.rumor_detect

# 常用工具函数

## 获取环境变量
::: RumorDetect.component.get_env

## 下载并解压到指定目录
::: RumorDetect.component.check_and_download

## 获取默认下载目录
::: RumorDetect.component.get_default_path