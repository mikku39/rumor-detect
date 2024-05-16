from pathlib import Path
import http.client, json, urllib
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import requests
from bs4 import BeautifulSoup
import os
import shutil
from tqdm import tqdm

# 获取数据


def load_data(sentence):
    # 读取数据字典
    with open(f"{get_default_path()}/dict.txt", "r", encoding="utf-8") as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = "<unk>"
        data.append(int(dict_txt[s]))
    return data


def tx_search(keyword_list):
    conn = http.client.HTTPSConnection("apis.tianapi.com")  # 接口域名
    params = urllib.parse.urlencode(
        {"key": os.environ.get("TX_KEY"), "word": keyword_list[0]}
    )
    headers = {"Content-type": "application/x-www-form-urlencoded"}
    conn.request("POST", "/generalnews/index", params, headers)
    tianapi = conn.getresponse()
    result = tianapi.read()
    data = result.decode("utf-8")
    tx_data = json.loads(data)
    return tx_data


# 返回google搜索结果
def google_search(search_term, **kwargs):
    api_key = os.environ.get("CSE_API_KEY")
    cse_id = os.environ.get("CSE_ID")
    query_params = {"q": search_term, "key": api_key, "cx": cse_id}
    query_params.update(kwargs)
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1", params=query_params
    )
    google_data = response.json()
    if "items" in google_data:
        return google_data["items"]
    print(
        f"谷歌搜索查询失败。返回数据为：{google_data}，KEY为{api_key}, CSE_ID为{cse_id}"
    )
    return []


def get_url_ctx(url):
    conn = http.client.HTTPSConnection("apis.tianapi.com")  # 接口域名
    params = urllib.parse.urlencode({"key": os.environ.get("TX_KEY"), "url": url})
    headers = {"Content-type": "application/x-www-form-urlencoded"}
    conn.request("POST", "/htmltext/index", params, headers)
    tianapi = conn.getresponse()
    result = tianapi.read()
    data = result.decode("utf-8")
    news_data = json.loads(data)
    return news_data


def beauty_ctx(ctx):
    soup = BeautifulSoup(ctx, "lxml")
    return soup.get_text()


# 返回网页内容
def get_news_list(data_list):
    news_list = []
    for data in data_list:
        news_data = get_url_ctx(data["url"])
        if news_data["code"] == 200:
            news_list.append(
                (data["title"], data["url"], beauty_ctx(news_data["result"]["content"]))
            )
    return news_list


def data2np(input):
    data = load_data(input)
    data_np = np.array(data)
    data_np = (
        np.array(
            np.pad(
                data_np,
                (0, max(0, 150 - len(data_np))),
                "constant",
                constant_values=4409,
            )
        )
        .astype("int64")
        .reshape(-1)
    )

    if len(data_np) > 150:
        data_np = data_np[:150]

    infer_np_doc = to_variable(data_np)
    return infer_np_doc


def generate_data_source(sent, data_list):
    return [{"query": sent, "title": data[2], "label": 0} for data in data_list]


def generate_data(sent, data_list):
    return [[data[2], sent] for data in data_list]


# 根据操作系统确定默认的下载路径
def get_default_path():
    """
    根据操作系统确定默认的下载路径
    默认为$HOME/tmp
    """
    if os.name == "nt":  # Windows系统
        # 使用Pathlib处理路径，获取默认的下载目录
        download_path = Path.home() / "tmp"
    elif os.name == "posix":  # Unix-like系统，如Linux
        # 同样使用Pathlib处理路径
        download_path = Path.home() / "tmp"
    else:
        # 对于其他操作系统，也可以定义路径或抛出错误
        raise NotImplementedError("Unsupported operating system.")
    return download_path


def check_and_download(path, url):
    if not os.path.exists(path):
        print(f"文件 {path} 不存在，正在下载...")
        # 发送GET请求
        response = requests.get(url, stream=True)
        # 确保请求成功
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        if url.endswith(".zip"):
            with open(path + ".zip", "wb") as f, tqdm(
                desc=path + ".zip",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            print("解压中")
            shutil.unpack_archive(
                filename=path + ".zip", extract_dir=path, format="zip"
            )
        else:
            with open(path, "wb") as f, tqdm(
                desc=path,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
        print(f"下载完成，文件已保存到 {path}")
    else:
        print(f"文件 {path} 已存在，无需下载。")
