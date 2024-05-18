import os
import requests
import jieba.analyse
import shutil
from tqdm import tqdm
from pathlib import Path


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


def noop_init():
    pass


# 查找关键词
def get_keywords(sent):
    """
    通过 jieba.analyse.extract_tags 查找微博文本输出关键词列表。

    Args:
        row: 一个 str。
    """
    res = jieba.analyse.extract_tags(sent)
    return res
