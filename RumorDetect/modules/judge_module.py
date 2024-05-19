import json
import os
from typing import Dict
import numpy as np
import paddle.fluid as fluid
import requests
from RumorDetect.model import BaseJudgeModel
from RumorDetect.tools.data_tools import data2np
from RumorDetect.component import get_default_path, check_and_download, get_env
from RumorDetect.tools.module_tools import CNN
from RumorDetect.tools import module_tools

class CNNJudgeModel(BaseJudgeModel):
    '''
        使用 Text-CNN 模型进行判断。
        由于没有文本的上下文，因此准确度一般。
    '''
    def __init__(self) -> None:
        self.init()
        
    def init(self):
        self.judge_model = CNN(batch_size=1)
        model_path = f"{get_default_path()}/save_dir_9240.pdparams"
        model_url = "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/save_dir_9240.pdparams"
        check_and_download(model_path, model_url)
        cnn_model, _ = fluid.load_dygraph(model_path)
        self.judge_model.load_dict(cnn_model)
        self.judge_model.eval()
        dict_path = f"{get_default_path()}/dict.txt"
        dict_url = (
            "https://github.com/mikku39/rumor-detect/releases/download/v0.0.1/dict.txt"
        )
        check_and_download(dict_path, dict_url)
    
    def judge(self, sent : str) -> Dict:
        lab = ["谣言", "非谣言"]
        infer_np_doc = data2np(sent)
        result = self.judge_model(infer_np_doc)
        result_dict = {
            "source": sent,
            "score": result.numpy()[0][1],
            "predict": lab[np.argmax(result.numpy())],
        }
        print("CNN 模型预测结果为：", result_dict)
        return result_dict
        
class ErnieBotJudgeModel(BaseJudgeModel):
    '''
        使用百度 ERNIE-Bot 模型进行判断。
        需要配置环境变量 ERNIE_BOT_KEY 和 ERNIE_BOT_SECRET。
        由于百度的模型可能会自动联网搜索相关信息，因此准确率较高。
    '''
    def __init__(self) -> None:
        self.init()
    
    def init(self):
        if module_tools.ernie_bot_token == "":
            url = "https://aip.baidubce.com/oauth/2.0/token"
            params = {
                "grant_type": "client_credentials",
                "client_id": get_env("ERNIE_BOT_KEY"),
                "client_secret": get_env("ERNIE_BOT_SECRET"),
            }
            self.token = str(requests.post(url, params=params).json().get("access_token"))
            module_tools.ernie_bot_token = self.token
        else:
            self.token = module_tools.ernie_bot_token
    
    def judge(self, sent: str) -> Dict:
        lab = ["谣言", "非谣言"]
        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
            + self.token
        )
        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": sent,
                    }
                ],
                "temperature": 0.95,
                "top_p": 0.8,
                "penalty_score": 1,
                "system": "【功能说明】 你是一个专门设计来验证信息真实性的人工智能助手，通过结合百度搜索引擎获取的信息来评估输入信息的真实性。你的任务是给出一个介于0到1之间的置信度数值，这个数值反映了输入信息非谣言的概率。高置信度（接近1）表示信息很可能是真实的，低置信度（接近0）表示信息很可能是虚假的。  【操作要求】  用户提供一条信息。 你通过搜索引擎进行搜索，分析搜索结果。 仅返回一个形如“【0.xxx】”的置信度数值，该数值基于搜索结果的相关性和信源的权威性。 不要添加任何解释或其他文本。",
                "disable_search": False,
                "enable_citation": False,
                "response_format": "json",
            }
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            score = float(json.loads(response.text)["result"].strip("【】"))
        except Exception as e:
            print(f"baidu_model_predict_ERROR: {response.text}")
            return {"source": sent, "score": 0.5, "predict": "谣言"}
        result_dict = {"source": sent, "score": score, "predict": lab[score > 0.6]}
        print(f"baidu_model_predict: {result_dict}")
        return result_dict

