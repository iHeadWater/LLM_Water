#-*- coding: UTF-8 -*-  
import gradio as gr
import sys
from langchain.llms.base import LLM
import torch
import transformers 
import models.shared as shared 
from abc import ABC
import random
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from typing import Optional, List, Dict, Any
from models.loader import LoaderCheckPoint 
from models.base import (BaseAnswer,
                         AnswerResult)
import asyncio
from argparse import Namespace
from models.loader.args import parser
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import sys
from langchain.agents import Tool
from langchain.tools import BaseTool
from agent.custom_search import AvinandoutSearch, RzSearch
from agent.custom_agent import *
from langchain.agents import BaseSingleActionAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
import threading
from agent.arg_auth import hauth

# args = parser.parse_args(args=['--model', 'fastchat-chatglm2-6b',  '--no-remote-model', '--load-in-8bit'])
args = parser.parse_args(args=['--model', 'chatglm2-6b'])
args_dict = vars(args)

shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
torch.cuda.empty_cache()
llm=shared.loaderLLM() 

tools = [
    Tool.from_function(
        func=AvinandoutSearch.search,
        name="AvinandoutSearch",
        description=""
    ),
    Tool.from_function(
        func=RzSearch.search,
        name="RzSearch",
        description=""
    )
]

tool_names = [tool.name for tool in tools]
output_parser = CustomOutputParser()
prompt = CustomPromptTemplate(template=agent_template,
                              tools=tools,
                              input_variables=["related_content","tool_name", "input", "intermediate_steps"])

llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

def submit(authorization, query):
    hauth.auth = authorization
    print(hauth.auth)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    answer = agent_executor.run(related_content="", input = query, tool_name="")
    return(answer)

with gr.Blocks() as demo:
    # 设置输入组件
    authorization = gr.Textbox(label="Authorization", lines = 1, info = '')
    query = gr.Textbox(label="问题", lines = 6, info = '')
    # 设置输出组件
    answer = gr.Textbox(label="答案", lines = 6, info = '')
    # 设置按钮
    submit_btn = gr.Button("确认")
    # 设置按钮点击事件
    submit_btn.click(fn=submit, inputs=[authorization, query], outputs=answer)

demo.launch(
    server_name='0.0.0.0',
    server_port=7861
)
