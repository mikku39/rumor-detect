import os
import click
from flask import Flask, jsonify, render_template, request


app = Flask(__name__, template_folder='/home/mikku/rumor-detect/RumorDetect/cmd/templates')

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
    result = instance.run(input_text)
    # return jsonify({'message': 'Hello from Flask!'})
    return jsonify({'message': 'Hello from Flask!', 'result': result})

@click.group()
def cli():
    """Group for the command line tool."""
    print("CLI Current Working Directory:", os.getcwd())

@cli.command()
def serve():
    """Command to start the Flask server."""
    print("Serving from:", os.getcwd())
    app.run(debug=False, port=5000)

if __name__ == '__main__':
    cli()
