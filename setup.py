from setuptools import setup, find_packages

setup(
    name="RumorDetect",
    version="1.0",
    description="YKK ysu grad project",
    author="mikku",
    author_email="mikku39chr@gmail.com",
    packages=find_packages(),
    package_data={
        'RumorDetect': ['cmd/templates/*.html'],  # Include all HTML files in cmd/templates
    },
    install_requires=[
        # "paddlepaddle==2.3.2",
        "paddlepaddle-gpu==2.3.2",
        "paddlehub==1.7.1",
        "python-dotenv",
        # "mkdocs",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "pymdown-extensions",
        "mkdocs-material",
        "mkdocstrings",
        "mkdocstrings[python]",
        "tabulate",
        "jieba",
        "url2io_client",
        "requests==2.31.0",
        "pre_commit==3.5.0",
        "click==8.1.7",
        "MarkupSafe==2.1.5",
        "zipp==3.17.0",
        "flask",
    ],
    entry_points={
        "console_scripts": [
            "RumorDetect = RumorDetect.cmd.serve:cli",
        ],
    },
    # scripts=[
    #     "cmd/bin/install_algo_pipeline_runner_configs",
    # ],
)