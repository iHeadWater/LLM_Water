# A simple introduction to LLM_Water

LLM_Water is an application mainly based on [LangChain_ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM/tree/master). It mainly focus on the QA function, based on knowledge base about water conservancy. For now, there aren't enough changes from Langchain_ChatGLM. So if you'd like to learn about how it works logically, I suggest you click the open link above. This README file is mainly focus on the way to use it and the work I need to go on.
# Install

Make sure your Python version is at least higher than 3.8.

```python
python --version
# Python 3.10.11
```
If your version is less than Python 3.8, you need to update it or create a virtual environment by Anaconda.

```python
conda create -p /your_path/env_name python=3.8

# activate
conda activate /your_path/env_name

# install
git clone https://github.com/imClumsyPanda/langchain-ChatGLM.git
cd langchain-ChatGLM

Install packages.
```python
pip install -r requirements.txt
# The packages offered by LangChain to deal with unstructed text
pip install "unstructured[local-inference]"
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
pip install layoutparser[layoutmodels,tesseract]
```

# Optional
Milvus can be used to replace FAISS as the vector store.

Here are the ways to install docker and docker-compose.
And if you have podman on your server, you can use podman and podman-compose to replace docker.

## Run Milvus
Get docker-compose file.
```python
wget https://github.com/milvus-io/milvus/releases/download/v2.2.10/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
Run.
```python
sudo docker-compose up -d
sudo docker-compose ps
sudo docker-compose down
```

If you'd like to run with podman, use this.
```python
mv docker-compose.yml podman-compose.yml
sudo podman-compose up -d
sudo podman-compose ps
sudo podman-compose down
```


# How To Use
First, check configs/model_config.py to make sure your server is capable to run.
Model     | Minimum GPU Memory| Minimum GPU Memory for Finetune
-------- | -----|------
chatglm-6b  | 13GB| 14GB
chatglm-6b-int4  | 8GB| 9GB
chatglm-6b-int8  | 6GB|7GB
chatglm-6b-int4-qe  | 6GB|7GB
moss|68GB|-
chatyuan|-|-

Besides, embedding model also needs about 3GB GPU Memory, or you can change it to CPU mode.
If you find out that your GPU Memory is not capable to run, please change the args in configs/model_config.py and models/loader/args.py.
```python
#model_config.py
LLM_MODEL = "chatglm-6b"

#args.py
parser.add_argument('--model', type=str, default='chatglm-6b', help='Name of the model to load by default.')
```
And if you'd like to run these model locally, you can follow the steps below to download the models.

```python
# install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs
git lfs install

# download the model
git clone https://huggingface.co/THUDM/chatglm-6b
```
If the speed is too slow to download from huggingface, try this.
```python

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b
```
Then download the download the checkpoints from [here](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/). And change the llm_model_dict path and the embedding_model_dict path in configs/model_config.py.

Run the demo.
```Python
python cli_demo.py
```
Run the webui.py
```Python
python webui.py

#output
Running on local URL:  http://0.0.0.0:7861

To create a public link, set `share=True` in `launch()`.
```
You can open the link if you are running loaclly.
Or if you are running on a server, you need to open the port 7861 using the firewall. And you can enter the link and find out the page below.
![在这里插入图片描述](https://img-blog.csdnimg.cn/740d51f9cb824c9b9df484f2e4f29249.png)
Just follow the instruction step by step. And you can get an easy QA application based on knowledge base. If you are a member of our team, you can read this [file](https://dlut-water.yuque.com/kgo8gd/tnld77/pydd7sgc05g470n0) to get a detailed instruction.

The documents you upload will be stored in content folder. And the processed index for faiss will be stored in vector_store.

# To Do
Create and combine Agents and Chains/Tools
Test the accuracy when there are a large amount of documents.
Accelerate the generation of ChatGLM
