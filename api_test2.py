import argparse
import json
import os
import shutil
from typing import List, Optional, Union, Dict
import urllib

import nltk
import pydantic
import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse

from chains.local_doc_qa import LocalDocQA
from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import requests

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")



def get_user_id(token: str = Depends(oauth2_scheme)):
    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Authorization':"Bearer " + token
    }
    # print(headers)
    url = "https://platform.waterism.com:8155/admin/user/info"

    user_id = requests.get(url, headers=headers).json()

    return(user_id['data']['sysUser']['username'])

def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")

class ChatResponse(BaseModel):
    code: int = pydantic.Field(0, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")
    data: Dict[str, Union[str, List[List[str]], List[str], List[Dict[str, Union[int, str]]]]] = pydantic.Field({
        "knowledge_base_id": "Biliu",
        "question": "碧流河水库调度方案？",
        "response": "根据已知信息，碧流河的调度方案如下：……………………",
        "history": [
            [
                "碧流河在哪里？",
                "根据已知信息，碧流河的所在地是辽宁省普兰店市、庄河市和盖州市的交界处，距大连市区175千米。",
            ]
        ],
        "source_documents": [
            "出处 [1] ...",
            "出处 [2] ...",
            "出处 [3] ...",
        ]},
        description="List of document names")
    class Config:
        schema_extra = {
            "example": {
                "code": 0,
                "msg": "success",
                "data": {
                    "knowledge_base_id": "Biliu",
                    "question": "碧流河水库调度方案？",
                    "response": "根据已知信息，碧流河的调度方案如下：……………………",
                    "history": [
                        [
                            "碧流河在哪里？",
                            "根据已知信息，碧流河的所在地是辽宁省普兰店市、庄河市和盖州市的交界处，距大连市区175千米。",
                        ]
                    ],
                    "source_documents": [
                        "出处 [1] ...",
                        "出处 [2] ...",
                        "出处 [3] ...",
                    ],
                }
            }
        }
        
async def chat(
    # knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
    token: str = Depends(oauth2_scheme),
    question: str = Body(..., description="Question", example="碧流河调度方案是什么？"),
    history: List[List[str]] = Body(
        [],
        description="History of previous questions and answers"
    )
    ):
    print(token)
    user_id:str = get_user_id(token)
    knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="Biliu")
    knowledge_base_id = "Biliu"
    vs_path = get_vs_path(str(knowledge_base_id))
    source_documents = []
    if not os.path.exists(vs_path):
        # return ChatResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        response=f"Knowledge base {knowledge_base_id} not found"
        all_data = {
            "knowledge_base_id": "Biliu",
            "question": question,
            "response": response,
            "history": history,
            "source_documents": source_documents
        }
        data_store = {
            "user_id":user_id,
            "prompt": question,
            "response": response,
            "history": history
        }
        with open('./dataset/user_dataset.json','a+',encoding = 'utf-8') as f:
            json.dump(data_store, f, ensure_ascii=False)
            f.write("\n")
            f.close
        return ChatResponse(code = 300, msg = "failed", data = all_data)
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]
        response=resp["result"]
        all_data = {
            "knowledge_base_id": "Biliu",
            "question": question,
            "response": response,
            "history": history,
            "source_documents": source_documents
        }
        data_store = {
            "user_id": user_id,
            "prompt": question,
            "response": response,
            "history": history
        }
        with open('./dataset/user_dataset.json','a+',encoding = 'utf-8') as f:
            json.dump(data_store, f, ensure_ascii=False)
            f.write("\n")
            f.close
        return ChatResponse(code = 0, msg = "success", data = all_data)

def api_start(host, port):
    global app
    global local_doc_qa

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    app = FastAPI()

    app.post("/chat", response_model=ChatResponse)(chat)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7900)
    # 初始化消息
    args = None
    args = parser.parse_args(args=['--model', 'chatglm2-6b', '--no-remote-model'])
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    api_start(args.host, args.port)
