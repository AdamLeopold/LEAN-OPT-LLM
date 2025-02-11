from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
import openai
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datetime import datetime, time


v1 = pd.read_csv('/Users/adam/SQ_dataset/AttractiveValues.csv', index_col=None)
v2 = pd.read_csv('/Users/adam/SQ_dataset/adjusted_AttractiveValues.csv', index_col=None)
v1 = v1.set_index(v1.columns[0])
v2 = v2.set_index(v2.columns[0])
v_br = v2/v1
v_br = v_br.reset_index()
print(v_br)


def ticket_analysis(ticket_info: str):
    """当你需要根据检索出的ticket信息，判断这些ticket的POS、departure_time区间、Oneway_Product，并从v_br中查找对应值时，请使用此工具。"""
    global v1

    # 初始化用于存储值的列表
    product_values = []
    no_purchase_values = []

    # 分割 ticket_info 字符串，获取每个 ticket 的信息
    tickets = ticket_info.strip().split(';')
    for ticket in tickets:
        # 提取 POS、departure_time、Oneway_Product
        pos_match = re.search(r'POS\s*=\s*([A-Z])', ticket)
        time_match = re.search(r'departure_time\s*[t\d]*\s*=\s*([\d:]+)', ticket)
        product_match = re.search(r'Oneway_product\s*[a\d]*\s*=\s*(\w+)', ticket)

        if not pos_match or not time_match or not product_match:
            pos_match = re.search(r'POS\s*=\s*([A-Z])', ticket, re.IGNORECASE)
            time_match = re.search(r'Departure\s*Time\s*=\s*([\d:]+)', ticket, re.IGNORECASE)
            product_match = re.search(r'Product\s*=\s*(\w+)', ticket, re.IGNORECASE)

        print(pos_match.group(1))
        print(time_match.group(1))
        print(product_match.group(1))

        pos = pos_match.group(1)
        departure_time_str = time_match.group(1)
        product = product_match.group(1)

        # 转换 departure_time_str 为时间对象
        departure_time = datetime.strptime(departure_time_str, '%H:%M').time()

        # 定义时间区间
        intervals = {
            '12pm~6pm': (time(12, 0), time(18, 0)),
            '6pm~10pm': (time(18, 0), time(22, 0)),
            '10pm~8am': (time(22, 0), time(8, 0)),
            '8am~12pm': (time(8, 0), time(12, 0))
        }

        # 确定 departure_time 所属的时间区间
        time_interval = None
        for interval_name, (start_time, end_time) in intervals.items():
            if start_time <= end_time:
                if start_time <= departure_time < end_time:
                    time_interval = interval_name
                    break
            else:
                if departure_time >= start_time or departure_time < end_time:
                    time_interval = interval_name
                    break

        if time_interval is None:
            continue  # 如果没有匹配的时间区间，跳过该 ticket

        # 构造 v_br 的列名
        key = product + '*(' + time_interval + ')'

        # 从 v_br 中查找对应的值
        subset = v1[v1['POS'] == pos]
        value = subset[key].values[0]
        product_values.append(value)

        # 添加 no_purchase 值（每个 POS 只添加一次）
        no_purchase_value = subset['no_purchase'].values[0]
        if no_purchase_value not in no_purchase_values:
            no_purchase_values.append(no_purchase_value)

    # 合并产品值和 no_purchase 值
    result_values = product_values + no_purchase_values

    # 返回结果列表
    return result_values

# 将新工具添加到tools列表中
# tools.append(ticket_analysis)

s="""
POS = A, departure_time t1 = 7:40, Oneway_product a1 = Eco_flexi, avg_pax d1 = 5.351, avg_price p1 = 1630.291;
POS = A, departure_time t2 = 7:40, Oneway_product a2 = Eco_lite, avg_pax d2 = 1.488, avg_price p2 = 483.509.
"""
result=ticket_analysis(s)
print(result)
