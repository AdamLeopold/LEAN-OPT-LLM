from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain import LLMChain
from langchain.prompts import PromptTemplate

import openai
import time
import re
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.vectorstores import Chroma

import os
import pandas as pd


import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_PROJECT'] = "or_expert"


user_api_key="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"
llm = ChatOpenAI(temperature=0.0, model_name='gpt-4', openai_api_key=user_api_key)


information=pd.read_csv('nike Shoes Sales.csv')
information_head=information[:36]


# 将示例数据转换为字符串，供 few_shot_examples 使用
example_data_description = "\nHere is the product data:\n"
for index, row in information_head.iterrows():
    example_data_description += f"Product {index+1}: {row['Product Name']}, revenue w_{index+1} = {row['Revenue']}, demand rate a_{index+1} = {row['Demand']}, initial inventory c_{index+1} = {row['Initial Inventory']}\n"

# 构建 problem_description 和 label
problem_description0 = problem_description = """The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer."""

problem_description += example_data_description


problem_description1='The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories of the Nike x Olivia Kim brand, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer.'

label_head1="""
Maximize
   11197 x_0 + 9097 x_1 + 11197 x_2 + 9995 x_3
Subject To
 inventory_constraint: x_0 <= 97
 demand_constraint: x_0 <= 17
 x_1 <= 240
 x_1 <= 26
 x_2 <= 322
 x_2 <= 50
 x_3 <= 281
 x_3 <= 53
"""

few_shot_examples = f"""

Question: {problem_description1}

Based on the above description and data, please formulate a linear programming model.

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file.

Action: CSVQA

Action Input: "Retrieve the product data related to Nike x OliviaKim to formulate the linear programming model."

Observation: {example_data_description}

Thought: Now that I have the necessary data, I can construct the objective function and constraints. And I should generate the answer only using the format below. The expressions should not be simplified or abbreviated. I need to retrieve products similar to 'Nike x Olivia Kim' from the CSV file to formulate the linear programming model.

Final Answer: 
{label_head1}
"""


# 加载实际的 CSV 文件
df = pd.read_csv("nike Shoes Sales.csv")

# 创建嵌入和向量存储
data = []
for index, row in df.iterrows():
    content = f"Product Name: {row['Product Name']}, Revenue: {row['Revenue']}, Demand: {row['Demand']}, Initial Inventory: {row['Initial Inventory']}"
    data.append(content)

documents = [content for content in data]
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
vectors = FAISS.from_texts(documents, embeddings)

num_documents = len(documents)

# 创建检索器和 RetrievalQA 链
retriever = vectors.as_retriever(search_kwargs={'k': 10})
llm = ChatOpenAI(temperature=0.0, model_name='gpt-4', openai_api_key=user_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
)

# 创建工具（Tool）
qa_tool = Tool(
    name="CSVQA",
    func=qa_chain.run,
    description="Use this tool to answer questions based on the provided CSV data and retrieve product data similar to the input query.."
)


# 更新 Agent 的提示（Prompt）
prefix = f"""You are an assistant that generates linear programming models based on the user's description and provided CSV data.

Please refer to the following example and generate the answer in the same format:

{few_shot_examples}

When you need to retrieve information from the CSV file, use the provided tool.

"""

suffix = """

Begin!

User Description: {input}
{agent_scratchpad}"""

# 初始化 Agent
agent = initialize_agent(
    tools=[qa_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={
        "prefix": prefix,
        "suffix": suffix,
    },
    verbose=True,
    handle_parsing_errors=True,
)

# 准备新的用户描述（不包含具体数据，让 Agent 自行检索）
user_description = 'The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories of the Kyrie brand, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer.'

# 运行 Agent 并获取答案
answer = agent.invoke(user_description)
print("Answer:", answer['output'])
