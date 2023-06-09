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
# In some servers, you may need to do "source activate base" first
source activate /your_path/env_name
pip3 install --upgrade pip

# deactivate
source deactivate /your_path/env_name

# remove
conda env remove -p  /your_path/env_name
```
Then, before pip the packages, you need to check if you have uninstalled detectron2. It might cause some conflicts.

```python
# check
pip show detectron2
pip uninstall detectron2

# install
git clone https://github.com/imClumsyPanda/langchain-ChatGLM.git
cd langchain-ChatGLM
```
Here are two packages recommended to be installed by LangChain-ChatGLM.
But actually I don't have the permission to use ‘yum’ in our company server, and this application still works well.

```python
yum install libX11
yum install libXext
```
Install packages.
```python
pip install -r requirements.txt
# The packages offered by LangChain to deal with unstructed text
pip install "unstructured[local-inference]"
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
pip install layoutparser[layoutmodels,tesseract]
```
It will be a little bit difficult to install the version that *requirements.txt* acquires. Even if this model is developed by Tsinghua University, you can't download the versions from  their mirror source. Just search these packages on the official website of the packages.

LangChain-ChatGLM also gives a way of docker installation.(I haven't tested it for the same reason, no root permission.)

```python
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit-base
sudo systemctl daemon-reload 
sudo systemctl restart docker

# online
docker build -f Dockerfile-cuda -t chatglm-cuda:latest .
docker run --gpus all -d --name chatglm -p 7860:7860  chatglm-cuda:latest

# offline
docker run --gpus all -d --name chatglm -p 7860:7860 -v ~/github/langchain-ChatGLM:/chatGLM  chatglm-cuda:latest
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
```

If you have mutiple GPU services and you'd like to run on specific GPU, try this.

```Python
nvidia-smi
CUDA_VISIBLE_DEVICES=1 python webui.py

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
Replace FAISS with Milvus
