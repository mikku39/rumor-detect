from contextlib import contextmanager
import os
import click
from flask import Flask, jsonify, render_template, request
import io
import sys
from flask_socketio import SocketIO, emit
import numpy as np

from RumorDetect.RumorDetect import rumor_detect
import re
import os

app = Flask(
    __name__, template_folder=f"{os.path.dirname(__file__)}/templates"
)
socketio = SocketIO(app)

clients = {}


@socketio.on("connect")
def on_connect():
    clients[request.sid] = request.sid  # Store session ID


@socketio.on("disconnect")
def on_disconnect():
    clients.pop(request.sid, None)  # Remove session ID


enable_debug = False
instance = rumor_detect(
    # 参数配置
    news_mode=["bing"],
    summary_mode=["ernie_bot"],
    compare_mode=["ernie_bot", "match"],
    judge_mode=["cnn", "ernie_bot"],
    auto_init=False,
)
debug_instance = None
list_list = [
    instance.list_available_keywords_mode,
    instance.list_available_news_mode,
    instance.list_available_summary_mode,
    instance.list_available_compare_mode,
    instance.list_available_judge_mode,
]

get_list = [
    instance.get_keywords_mode,
    instance.get_news_mode,
    instance.get_summary_mode,
    instance.get_compare_mode,
    instance.get_judge_mode,
]

set_list = [
    instance.set_keywords_mode,
    instance.set_news_mode,
    instance.set_summary_mode,
    instance.set_compare_mode,
    instance.set_judge_mode,
]


@app.route("/list", methods=["POST"])
def list():
    data = request.json
    index = 0
    if "index" in data:
        index = data["index"]
    return jsonify(list_list[index]())


@app.route("/get", methods=["POST"])
def get():
    data = request.json
    index = 0
    if "index" in data:
        index = data["index"]
    return jsonify(get_list[index]())


@app.route("/delete", methods=["POST"])
def delete():
    # 处理接收到的数据，例如删除标签或处理选项
    data = request.json
    if "delete_tag" in data and "index" in data:
        delete_tag = data["delete_tag"]
        index = data["index"]
        tmp_list = get_list[index]()
        set_list[index]([i for i in tmp_list if i != delete_tag])
    return jsonify({"success": True})


@app.route("/submit", methods=["POST"])
def submit():
    # 处理接收到的数据，例如删除标签或处理选项
    data = request.json
    if "selectedValue" in data and "index" in data:
        selectedValue = data["selectedValue"]
        index = data["index"]
        tmp_list = get_list[index]()
        tmp_list.append(selectedValue)
        set_list[index](tmp_list)
    return jsonify({"success": True})


class StreamToSocketIO(io.StringIO):
    def __init__(self, emit_event_name, namespace="/", room=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emit_event_name = emit_event_name
        self.namespace = namespace
        self.room = room

    def write(self, value):
        super().write(value)
        # Emit the output in real-time to clients
        # Make sure to emit with the correct namespace and room
        if self.room:
            socketio.emit(
                self.emit_event_name,
                {"data": value},
                namespace=self.namespace,
                room=self.room,
            )
        else:
            socketio.emit(
                self.emit_event_name, {"data": value}, namespace=self.namespace
            )

    def flush(self):
        pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data", methods=["POST"])
def get_data():
    global debug_instance
    data = request.json
    input_text = data["text"]  # 假设传入的文本存储在 'text' 键中
    original_stdout = sys.stdout  # Keep track of the original stdout
    if not enable_debug:
        print(input_text)
        try:
            # Replace sys.stdout with our custom stream
            sys.stdout = StreamToSocketIO("output")
            result = instance.run(input_text)
        finally:
            sys.stdout = original_stdout  # Restore the original stdout
        return jsonify(result)
    else:
        if debug_instance is None:
            debug_instance = instance.debug_run(input_text)
        try:
            sys.stdout = StreamToSocketIO("output")
            print("单模块运行中...")
            next(debug_instance)
        except StopIteration:
            print("本次运行结束")
            debug_instance = None
        finally:
            print("单模块运行结束")
            sys.stdout = original_stdout  # Restore the original stdout
        print(f"aaaa:{result}")
        return jsonify(result)


@app.route("/intermediate")
def get_intermediate():
    result = instance.get_intermediate()
    return jsonify(
        {
            "message": "Done!",
            "sent": result["sent"],
            "keywords": result["keywords"],
            "news_list": result["news_list"],
        }
    )


@app.route("/debug_update", methods=["POST"])
def debug_update():
    data = request.json
    sent = data["sent"]
    keywords = data["keywords"]
    news_list = data["news_list"]
    keywords_list = re.split(r"[ ,]+", keywords)
    instance.update_params(
        {"sent": sent, "keywords": keywords_list, "news_list": news_list}
    )
    return jsonify({"message": "Done!"})

@click.group()
def cli():
    """Group for the command line tool."""
    pass


@cli.command()
def serve():
    """Command to start the Flask server."""
    print(os.path.dirname(__file__))
    print(f"{os.path.dirname(__file__)}/templates")
    socketio.run(app, debug=False, port=5000)


@cli.command()
def debug():
    global enable_debug
    enable_debug = True
    socketio.run(app, debug=False, port=5000)


if __name__ == "__main__":
    print(os.path.dirname(__file__))
    print(f"{os.path.dirname(__file__)}/templates")
    cli()
