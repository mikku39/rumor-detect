import json
import os
from typing import List, Tuple

import requests
from RumorDetect.model import BaseSummaryModel
from RumorDetect.component import check_and_download, get_default_path
from paddlenlp.transformers import AutoModelForConditionalGeneration
from paddlenlp.transformers import AutoTokenizer
from RumorDetect.tools import module_tools


class PegasusSummaryModel(BaseSummaryModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        path = f"{get_default_path()}/pegasus_checkpoints"
        url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/pegasus_checkpoints.zip"
        check_and_download(path, url)
        self.summary_model = AutoModelForConditionalGeneration.from_pretrained(path)
        self.summary_tokenizer = AutoTokenizer.from_pretrained(path)

    def get_summary(self, sent : str, news_list:List[Tuple]) -> Tuple[str, List[Tuple]]: 
        summary_list = []
        for news in news_list:
            text = news[2]
            summary_text = self.single_infer(text)
            summary_list.append((news[0], news[1], summary_text))

        if len(sent) > 20:
            sent = self.single_infer(sent)
        return sent, summary_list
    
    def single_infer(self, text : str )->str:
        # 针对单条文本，生成摘要
        tokenized = self.summary_tokenizer(
            text[:128], truncation=True, max_length=128, return_tensors="pd"
        )
        preds, _ = self.summary_model.generate(
            input_ids=tokenized["input_ids"],
            max_length=64,
            min_length=0,
            decode_strategy="beam_search",
            num_beams=4,
        )
        summary_ans = self.summary_tokenizer.decode(
            preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return summary_ans
    

class BaiduSummaryModel(BaseSummaryModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": os.environ.get("BAIDU_API_KEY"), "client_secret": os.environ.get("BAIDU_API_SECRET")}
        self.token = str(requests.post(url, params=params).json().get("access_token"))
        

    def get_summary(self, sent : str, news_list:List[Tuple]) -> Tuple[str, List[Tuple]]: 
        summary_list = []
        for news in news_list:
            text = news[2]
            summary_text = self.single_infer(text)
            summary_list.append((news[0], news[1], summary_text))

        if len(sent) > 20:
            sent = self.single_infer(sent)

        return sent, summary_list
    
    def single_infer(self, text : str )->str:
        url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/news_summary?charset=UTF-8&access_token=" + self.token
        payload = json.dumps({
            "content": text,
            "max_summary_len": 50
        })
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        result = response.json()
        if "error_code" in result:
            print(f"使用百度短文本比较的时候出错：{result}")
            return 
        return result["summary"]
    
class ErnieBotSummaryModel(BaseSummaryModel):
    def __init__(self) -> None:
        self.init()

    def init(self):
        if module_tools.ernie_bot_token == "":
            print("ccccccccccccccccccc")
            url = "https://aip.baidubce.com/oauth/2.0/token"
            params = {"grant_type": "client_credentials", "client_id": os.environ.get("ERNIE_BOT_KEY"), "client_secret": os.environ.get("ERNIE_BOT_SECRET")}
            self.token = str(requests.post(url, params=params).json().get("access_token"))
            module_tools.ernie_bot_token = self.token
        else:
            self.token = module_tools.ernie_bot_token

    def get_summary(self, sent : str, news_list:List[Tuple]) -> Tuple[str, List[Tuple]]: 
        summary_list = []
        for news in news_list:
            text = news[2]
            summary_text = self.single_infer(text)
            summary_list.append((news[0], news[1], summary_text))
        if len(sent) > 20:
            sent = self.single_infer(sent)
        return sent, summary_list
    
    def single_infer(self, text : str )->str:
        url = os.environ["ERIBOT_URL"] + self.token
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": text[:1024],
                }
            ],
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1,
            "system": "【功能说明】 你是一个专为提供文本概述设计的人工智能助手。你的任务是仅从所给文本中提取信息，生成一个简短的概述。在处理过程中，不得搜索或引入任何超出原始文本范围的信息。  【操作要求】  仅从用户提供的文本中分析和提取信息。 生成的概述必须直接基于输入的文本，严禁使用任何非输入文本的信息。 概述不得超过50字，并且必须紧密关联原文内容。 【示例】 如果输入文本是：“全球变暖导致极端天气事件的频率和强度增加，科学家们呼吁更多的气候行动。” 你的输出应该是：“科学家因全球变暖导致极端天气增加呼吁气候行动。”",
            "disable_search": True,
            "enable_citation": False,
            "response_format": "text"
        })
        headers = {
            'Content-Type': 'application/json'
        }
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            result = json.loads(response.text)["result"]
        except Exception as e:
            print(f"使用ErnieBot进行summary的时候出错，截取前50个字符：{e}")
            return text[:50]
        return result