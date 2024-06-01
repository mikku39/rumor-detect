import os
import requests
import shutil
from tqdm import tqdm
from pathlib import Path

import logging
import sys

# 创建 logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建 stream handler，用于输出到stderr
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.WARNING)  # 设置此handler的日志级别为WARNING

# 创建 formatter，并设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# 将 handler 添加到 logger
logger.addHandler(stream_handler)

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


def check_and_download(path: str, url: str):
    '''
        将文件流式下载到path目录，如果文件已存在则跳过。如果文件是zip格式则解压。

        Args:
            path: 保存路径
            url: 下载链接
    '''
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

def get_env(key: str, default: str=None):
    '''
    返回环境变量，如果不存在且没有默认值则抛出异常。

    Args:
        key: 环境变量名
        default: 默认值
    '''
    res = os.environ.get(key)
    if res is None:
        res = default
        if res is None:
            raise EnvironmentError(f"环境变量 {key} 未设置")
    return res

def noop_init():
    pass


