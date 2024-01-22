
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain import PromptTemplate, LLMChain
from agent.custom_search import DeepSearch, AvinandoutSearch, RzSearch
from langchain.agents import BaseSingleActionAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from typing import List, Tuple, Any, Union, Optional, Type
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.base_language import BaseLanguageModel
import re

agent_template = """
你现在是一个{role}。这里是一些已知信息：
{related_content}
{background_infomation}
{question_guide}{input}

{answer_format}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        # 没有互联网查询信息
        if len(intermediate_steps) == 0:
            background_infomation = "\n"
            role = "傻瓜机器人"
            question_guide = "这是已知的回答模板。\n\
问题:尼尔基水库当前的日均出入水量是多少？\n\
答案:AvinandoutSearch('尼尔基水库')\n\
问题:尼尔基水库现在的出入水量是多少？\n\
答案:AvinandoutSearch('尼尔基水库')\n\
问题:尼尔基水库现在的库水位？\n\
答案:RzSearch('尼尔基水库')\n\
问题:尼尔基水库当前的库水位？\n\
答案:RzSearch('尼尔基水库')\n\
模板结束。\n\
我现在有一个问题。\n\
问题："
            answer_format = "如果你知道答案，请直接给出你的回答！\n\
            如果你不知道答案，请你只回答\"AvinandoutSearch('搜索词')\"，\n\
            或者\"RzSearch('搜索词')\"，\\n\
            并将'搜索词'替换为你认为需要搜索的关键词，除此之外不要回答其他任何内容。\n\n下面请回答我上面提出的问题！"
            # answer_format = "请调用工具tool_a回答我的问题！下面回答我的问题！"

        # 返回了背景信息
        else:
            # 根据 intermediate_steps 中的 AgentAction 拼装 background_infomation
            background_infomation = "\n\n你还有这些已知信息作为参考：\n\n"
            print(intermediate_steps)
            action, observation = intermediate_steps[0]
            background_infomation += f"{observation}\n"
            role = "聪明的 AI 助手"
            question_guide = "请根据这些已知信息回答我的问题:"
            answer_format = ""

        kwargs["background_infomation"] = background_infomation
        kwargs["role"] = role
        kwargs["question_guide"] = question_guide
        kwargs["answer_format"] = answer_format
        return self.template.format(**kwargs)

'''
class CustomSearchTool(BaseTool):
    name: str = "AvinandoutSearch"
    description: str = ""

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        return AvinandoutSearch.search(query = query)

    async def _arun(self, query: str):
        raise NotImplementedError("AvinandoutSearch does not support async")


class CustomAgent(BaseSingleActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermedate_steps: List[Tuple[AgentAction, str]],
            **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        return AgentAction(tool="AvinandoutSearch", tool_input=kwargs["input"], log="")
'''

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # group1 = 调用函数名字
        # group2 = 传入参数
        print(llm_output)
        # search1 = re.search(r'^[\s\w]*(AvinandoutSearch)\(([^\)]+)\)', llm_output, re.DOTALL)
        search1 = re.search(r'(AvinandoutSearch)\(([^\)]+)\)', llm_output, re.DOTALL)
        # search2 = re.search(r'^[\s\w]*(RzSearch)\(([^\)]+)\)', llm_output, re.DOTALL)
        search2 = re.search(r'(RzSearch)\(([^\)]+)\)', llm_output, re.DOTALL)
        if not search1 and not search2 :
            print("not search")
        # 如果 llm 没有返回 AvinandoutSearch() 或者 RzSearch()则认为直接结束指令
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        # 否则的话都认为需要调用 Tool
        else:
            print("search")
            if search1:
                print("search1")
                action = search1.group(1).strip()
                action_input = search1.group(2).strip()
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
            else:
                print("search2")
                action = search2.group(1).strip() 
                action_input = search2.group(2).strip()
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        

class DeepAgent:
    tool_name: str = "AvinandoutSearch"
    agent_executor: any
    tools: List[Tool]
    llm_chain: any
    '''
    def query(self, related_content: str = "", query: str = ""):
        tool_name = self.tool_name
        result = self.agent_executor.run(related_content=related_content, input=query ,tool_name=self.tool_name)
        return result
    '''

    def __init__(self, llm: BaseLanguageModel, **kwargs):
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
        self.tools = tools
        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        prompt = CustomPromptTemplate(template=agent_template,
                                      tools=tools,
                                      input_variables=["related_content","tool_name", "input", "intermediate_steps"])

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.llm_chain = llm_chain

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        self.agent_executor = agent_executor

