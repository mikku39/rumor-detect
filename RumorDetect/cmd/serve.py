from contextlib import contextmanager
import os
import click
from flask import Flask, jsonify, render_template, request
import io
import sys
from flask_socketio import SocketIO, emit
import numpy as np



app = Flask(__name__, template_folder='/home/mikku/rumor-detect/RumorDetect/cmd/templates')
socketio = SocketIO(app)
clients = {}

@socketio.on('connect')
def on_connect():
    clients[request.sid] = request.sid  # Store session ID

@socketio.on('disconnect')
def on_disconnect():
    clients.pop(request.sid, None)  # Remove session ID
    
class StreamToSocketIO(io.StringIO):
    def __init__(self, emit_event_name, namespace='/', room=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emit_event_name = emit_event_name
        self.namespace = namespace
        self.room = room

    def write(self, value):
        super().write(value)
        # Emit the output in real-time to clients
        # Make sure to emit with the correct namespace and room
        if self.room:
            socketio.emit(self.emit_event_name, {'data': value}, namespace=self.namespace, room=self.room)
        else:
            socketio.emit(self.emit_event_name, {'data': value}, namespace=self.namespace)

    def flush(self):
        pass




@app.route('/')
def home():
    print("Home directory:", os.getcwd())
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def get_data():
    print("API directory:", os.getcwd())
    # 假设 RumorDetect 是你的类
    from RumorDetect.RumorDetect import rumor_detect
    # 创建 RumorDetect 的实例
    data = request.json
    input_text = data['text']  # 假设传入的文本存储在 'text' 键中
    instance = rumor_detect(
        # 参数配置
        news_mode=["bing"],
        summary_mode=["ernie_bot"],
        compare_mode=["ernie_bot", "match"],
        judge_mode=["cnn", "ernie_bot"],
    )
    sent = "狂飙兄弟没有在连云港遇到鬼秤"
    print(input_text)
    original_stdout = sys.stdout  # Keep track of the original stdout
    try:
        # Replace sys.stdout with our custom stream
        sys.stdout = StreamToSocketIO('output')
        result = instance.run(input_text)
    finally:
        sys.stdout = original_stdout  # Restore the original stdout
    return jsonify({'message': 'Done!'})

@click.group()
def cli():
    """Group for the command line tool."""
    print("CLI Current Working Directory:", os.getcwd())

@cli.command()
def serve():
    """Command to start the Flask server."""
    print("Serving from:", os.getcwd())
    socketio.run(app, debug=False, port=5000)

if __name__ == '__main__':
    cli()
