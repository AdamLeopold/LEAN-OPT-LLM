# 安装依赖项（注释掉以避免意外执行）
# pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
import re
import openai
import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.schema import Document
import ast
from datetime import time as time_class  
from datetime import datetime, time
from io import StringIO
import sys
from langchain.schema import HumanMessage

# 侧边栏配置区 ========================================================
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

# 文件上传组件（支持多个CSV文件）
uploaded_files = st.sidebar.file_uploader("upload", type="csv", accept_multiple_files=True)

if uploaded_files:
    dfs = []  # 用于存储读取的 DataFrames
    for uploaded_file in uploaded_files:
        # 使用临时文件处理上传内容（避免直接操作上传文件）
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 读取 CSV 文件并添加到列表中
        df = pd.read_csv(tmp_file_path)
        dfs.append((uploaded_file.name, df))

    # 大语言模型初始化 ==================================================
    llm1 = ChatOpenAI(
        temperature=0.0,    # 低随机性以保证确定性输出
        model_name="gpt-4o",
        openai_api_key=user_api_key
    )

    # 数据加载与向量化处理 ===============================================
    # 加载参考文档（RAG_doc2.csv）
    loader = CSVLoader(file_path="./Large_Scale_Or_Files/Ref_Data2_2.csv", encoding="utf-8")
    data = loader.load()

    # Each line is a document
    documents = data     # 每个CSV行视为一个文档

    # 创建嵌入模型和向量存储
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    vectors = FAISS.from_documents(documents, embeddings)

    # 创建检索器（返回前5个最相关结果）
    retriever_pc = vectors.as_retriever(search_kwargs={'k': 5})

    # 构建检索问答链 ====================================================
    qa_chain_pc = RetrievalQA.from_chain_type(
        llm=llm1,
        chain_type="stuff",   # 简单拼接上下文的方式
        retriever=retriever_pc,
        return_source_documents=True,   # 返回源文档用于调试
    )


    # Create a tool using the RetrievalQA chain  创建问答工具 ======================================================
    qa_tool_pc = Tool(
        name="FileQA",
        func=qa_chain_pc.invoke,
        description=(
            "Use this tool to answer questions about the problem type of the text. "
            "Provide the question as input, and the tool will retrieve the relevant information from the file and use it to answer the question."
        ),
    )

    # Define few-shot examples as a string  小样本示例（Few-shot Learning）配置 ================================
    few_shot_example_pc = """
    
    Question: What is the problem type in operation of the text? Please give the answer directly. Text:There are three best-selling items (P1, P2, P3) on Amazon with the profit w_1,w_2,w_3.There is an independent demand stream for each of the products. The objective of the company is to decide which demands to be fufilled over a ﬁnite sales horizon [0,10] to maximize the total expected revenue from ﬁxed initial inventories. The on-hand inventories for the three items are c_1,c_2,c_3 respectively. During the sales horizon, replenishment is not allowed and there is no any in-transit inventories. Customers who want to purchase P1,P2,P3 arrive at each period accoring to a Poisson process with a_1,a_2,a_3 the arrival rates respectively. Decision variables y_1,y_2,y_3 correspond to the number of requests that the firm plans to fulfill for product 1,2,3. These variables are all positive integers.
    
    Thought: I need to determine the problem type of the text. I'll use the FileQA tool to retrieve the relevant information.
    
    Action: FileQA
    
    Action Input: "What is the problem type in operation of the text? text:There are three best-selling items (P1, P2, P3) on Amazon with the profit w_1, w_2, w_3. ..."
    
    Observation: The problem type of the text is Network Revenue Management.
    
    Final Answer: Network Revenue Management.
    
    """

    # Create the prefix and suffix for the agent's prompt   Agent提示模板配置 =================================================
    prefix = f"""You are a helpful assistant that can answer questions about operation problems. 
    
    Use the following examples as a guide. Always use the FileQA tool when you need to retrieve information from the file:
    
    
    {few_shot_example_pc}
    
    When you need to find information from the file, use the provided tools. And answer the question by given the answer directly. For example,
    
    """

    suffix = """
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}"""

    # 初始化主Agent =====================================================
    agent_pc = initialize_agent(
        tools=[qa_tool_pc],
        llm=llm1,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,    # 零样本推理模式
        agent_kwargs={
            "prefix": prefix.format(few_shot_examples=few_shot_example_pc),
            "suffix": suffix,
        },
        verbose=True,     # 显示详细推理过程
        handle_parsing_errors=True,  # Enable error handling
    )

    # OpenAI API超时设置 ================================================
    openai.api_request_timeout = 60  # 将超时时间设置为60秒

    # 会话状态初始化 ====================================================
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True

    # 页面布局定义 ======================================================
    response_container = st.container()  # 响应展示区
    input_container = st.container()  # 输入区

    # 欢迎消息（仅首次显示）
    with response_container:
        if st.session_state.first_run:
            st.markdown(f"## Hello! Ask me anything about {uploaded_file.name} 🤗")
            st.session_state.first_run = False


    # 用户输入处理函数 ==================================================
    def process_user_input(query, agent_pc, container, dfs):

        # 记录用户输入
#        st.session_state.past.append(query)

        # 立即显示用户消息
        with container:
#            message(query, is_user=True, key=f"user_{len(st.session_state.past)}")
            message(query, is_user=True)

            # 第一步：问题分类 ===========================================
            with st.spinner(f'I would first analyze the problem and see which category it belongs to. Let me analyze...'):
                category_original=agent_pc.invoke(f"What is the problem type in operation of the text? text:{query}")

                # 定义正则表达式模式，匹配问题类型
                def extract_problem_type(output_text):
                    pattern = r'(Network Revenue Management|Resource Allocation|Transportation|Sales-Based Linear Programming)'
                    match = re.search(pattern, output_text, re.IGNORECASE)
                    return match.group(0) if match else None

                selected_problem = extract_problem_type(category_original['output'])

            # 第二步：具体问题处理 =========================================
            with st.spinner(f'I think this is a {selected_problem} Problem. Let me analyze...'):
                # 根据不同问题类型初始化不同Agent
                if selected_problem == "Network Revenue Management":

                    information = pd.read_csv('NRM_example/nike Shoes Sales.csv')
                    information_head = information[:36]

                    # 将示例数据转换为字符串，供 few_shot_examples 使用
                    example_data_description = "\nHere is the product data:\n"
                    for i, r in information_head.iterrows():
                        example_data_description += f"Product {i + 1}: {r['Product Name']}, revenue w_{i + 1} = {r['Revenue']}, demand rate a_{i + 1} = {r['Demand']}, initial inventory c_{i + 1} = {r['Initial Inventory']}\n"

                    problem_description = 'The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories of the Nike x Olivia Kim brand, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” Moreover, the trade will be started only when the total demand is no less than 100 to ensure the trading efficiency. The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer.'

                    label = """
                    Maximize
                       11197 x_1 + 9097 x_2 + 11197 x_3 + 9995 x_4
                    Subject To
                       inventory_constraint: 
                       x_1 <= 97
                       x_2 <= 240
                       x_3 <= 322
                       x_4 <= 281
                       demand_constraint: 
                       x_1 <= 17
                       x_2 <= 26
                       x_3 <= 50
                       x_4 <= 53
                       startup_constraint:
                       x_1 + x_2 + x_3 + x_4 >= 100
                    Where
                    x_i represents the number of customer requests the store intends to fulfill for Nike x Olivia Kim shoe style i, with each x_i being a positive integer.
    
                    """

                    few_shot_examples = f"""
    
                    Question: Based on the following description and data, please formulate a linear programming model. {problem_description}
    
                    Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file.
    
                    Action: CSVQA
    
                    Action Input: "Retrieve the product data related to Nike x OliviaKim to formulate the linear programming model."
    
                    Observation:
    
                    Thought: Now that I have the necessary data, I can construct the objective function and constraints. And I should generate the answer only using the format below. The expressions should not be simplified or abbreviated. I need to retrieve products similar to 'Nike x Olivia Kim' from the CSV file to formulate the linear programming model.
    
                    Final Answer: 
                    {label}
                    """

                    # 加载实际的 CSV 文件

                    # 创建嵌入和向量存储
                    data = []

                    for df_index, (file_name, df) in enumerate(dfs):
                        # 将文件名添加到描述中
                        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

                        for i, r in df.iterrows():
                            description = ""
                            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                            data.append(description + "\n")

    #                    print(data)


                    documents = [content for content in data]
                    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
                    vectors = FAISS.from_texts(documents, embeddings)

                    num_documents = len(documents)

                    # 创建检索器和 RetrievalQA 链
                    retriever = vectors.as_retriever(search_kwargs={'k': 250})
                    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4o', openai_api_key=user_api_key)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm2,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=False,
                    )

                    # 创建工具（Tool）
                    qa_tool = Tool(
                        name="CSVQA",
                        func=qa_chain.run,
                        description="Use this tool to answer questions based on the provided CSV data and retrieve product data similar to the input query."
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
                    agent2 = initialize_agent(
                        tools=[qa_tool],
                        llm=llm2,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        agent_kwargs={
                            "prefix": prefix,
                            "suffix": suffix,
                        },
                        verbose=True,
                        handle_parsing_errors=True,
                    )

                if selected_problem == "Resource Allocation":

                    informationC = pd.read_csv('RA_example/capacity.csv')
                    informationP = pd.read_csv('RA_example/products.csv')
                    information = []
                    information.append(informationC)
                    information.append(informationP)

                    # # 将示例数据转换为字符串，供 few_shot_examples 使用
                    example_data_description = "\nHere is the data:\n"

                    # 遍历每个 DataFrame 及其索引
                    for df_index, df in enumerate(information):
                        if df_index == 0:
                            example_data_description += f"\nDataFrame {df_index + 1} - Capacity\n"
                        elif df_index == 1:
                            example_data_description += f"\nDataFrame {df_index + 1} - Products\n"

                        # 遍历 DataFrame 的每一行并生成描述
                        for i, r in df.iterrows():
                            description = ""
                            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                            example_data_description += description + "\n"

                    # 构建 problem_description 和 label
                    problem_description = 'A supermarket needs to allocate various products, including high-demand items like the Sony Alpha Refrigerator, Sony Bravia XR, and Sony PlayStation 5, across different retail shelves. The product values and space requirements are provided in the "Products.csv" dataset. Additionally, the store has multiple shelves, each with a total space limit and specific space constraints for Sony and Apple products, as outlined in the "Capacity.csv" file. The goal is to determine the optimal number of units of each Sony product to place on each shelf to maximize total value while ensuring that the space used by Sony products on each shelf does not exceed the brand-specific limits. The decision variables x_ij represent the number of units of product i to be placed on shelf j.'

                    with open('RA_example/sony.txt', 'r', encoding='utf-8') as file:
                        label = file.read()

                    few_shot_examples = f"""
    
                            Question: Based on the following description and data, please formulate a linear programming model. {problem_description}
    
                            Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Particularly, I should fighure out which product and how many of them in total in the CSV should be considered.
    
                            Action: CSVQA
    
                            Action Input: "Retrieve the capacity data and products data to formulate the linear programming model."
    
                            Observation: {example_data_description}
    
                            Thought: Now that I have the necessary data, I can construct the objective function and constraints. And I should generate the answer only using the format below. The expressions should not be simplified or abbreviated. I need to retrieve products similar to 'Sony' from the CSV file to formulate the linear programming model.
    
                            Final Answer: 
                            {label}
                            """

                    # 加载实际的 CSV 文件
                    data = []
                    for df_index, (file_name, df) in enumerate(dfs):
                        # 将文件名添加到描述中
                        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

                        for i, r in df.iterrows():
                            description = ""
                            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                            data.append(description + "\n")

                    documents = [content for content in data]
                    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
                    vectors = FAISS.from_texts(documents, embeddings)

                    num_documents = len(documents)

                    # 创建检索器和 RetrievalQA 链
                    retriever = vectors.as_retriever(search_kwargs={'k': 220})
                    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4', openai_api_key=user_api_key)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm2,
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
                    agent2 = initialize_agent(
                        tools=[qa_tool],
                        llm=llm2,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        agent_kwargs={
                            "prefix": prefix,
                            "suffix": suffix,
                        },
                        verbose=True,
                        handle_parsing_errors=True,
                    )

                if selected_problem == "Transportation":

                    informationCustomer = pd.read_csv('./Large_Scale_Or_Files/TP_example/customer_demand.csv')
                    informationSupply = pd.read_csv('./Large_Scale_Or_Files/TP_example/supply_capacity.csv')
                    informationCost = pd.read_csv('./Large_Scale_Or_Files/TP_example/transportation_costs.csv')
                    information = []
                    information.append(informationCustomer)
                    information.append(informationSupply)
                    information.append(informationCost)

                    # # 将示例数据转换为字符串，供 few_shot_examples 使用
                    example_data_description = "\nHere is the data:\n"

                    # 遍历每个 DataFrame 及其索引
                    for df_index, df in enumerate(information):
                        if df_index == 0:
                            example_data_description += f"\nDataFrame {df_index + 1} - Customer Demand\n"
                        elif df_index == 1:
                            example_data_description += f"\nDataFrame {df_index + 1} - Supply Capacity\n"
                        elif df_index == 2:
                            example_data_description += f"\nDataFrame {df_index + 1} - Transportation Cost\n"

                        # 遍历 DataFrame 的每一行并生成描述
                        for i, r in df.iterrows():
                            description = ""
                            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                            example_data_description += description + "\n"

                    # 构建 problem_description 和 label

                    with open('./Large_Scale_Or_Files/TP_example/transportation_problem2.txt', 'r', encoding='utf-8') as file:
                        label = file.read()

                    few_shot_examples = f"""
    
                            Question: Based on the following transportation problem description and data, please formulate a complete linear programming model using real data from retrieval. {problem_description}
    
                            Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Particularly, I should fighure out which product and how many of them in total in the CSV should be considered.
    
                            Action: CSVQA
    
                            Action Input: Retrieve the capacity data and products data to formulate the linear programming model.
    
                            Observation:
    
                            Thought: Now that I have the necessary data, I can construct the objective function and constraints. And the answer I generate should only be similar to the format below. The expressions should not be simplified or abbreviated.
    
                            Final Answer: 
                            {label}
                            """

                    data = []
                    for df_index, (file_name, df) in enumerate(dfs):
                        # 将文件名添加到描述中
                        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

                        for i, r in df.iterrows():
                            description = ""
                            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                            data.append(description + "\n")

                    documents = [content for content in data]
                    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
                    vectors = FAISS.from_texts(documents, embeddings)

                    num_documents = len(documents)

                    # 创建检索器和 RetrievalQA 链
                    retriever = vectors.as_retriever(search_kwargs={'k': 250})
                    llm2 = ChatOpenAI(temperature=0.0, model_name='gpt-4', openai_api_key=user_api_key)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm2,
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
                    agent2 = initialize_agent(
                        tools=[qa_tool],
                        llm=llm2,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        agent_kwargs={
                            "prefix": prefix,
                            "suffix": suffix,
                        },
                        verbose=True,
                        handle_parsing_errors=True,
                    )


                    # 执行问题求解
                    result = agent2.invoke(query)
                    ai_response = result['output']

                if selected_problem == "Sales-Based Linear Programming":
                    def LoadFiles():
                        v1 = pd.read_csv('./Test_Dataset/Air_NRM/v1.csv')
                        v2 = pd.read_csv('./Test_Dataset/Air_NRM/v2.csv')
                        info = pd.read_csv('./Test_Dataset/Air_NRM/information.csv')
                        return v1,v2,info  
                    def New_Vectors(query):
                        v1,v2,df = LoadFiles()
                        new_docs = []
                        for _, row in df.iterrows():
                            new_docs.append(Document(
                                page_content=f"OD={row['OD']},Departure_Time_Flight1={row['Departure Time']},Oneway_Product={row['Oneway_Product']}, avg_pax={row['Avg Pax']}, avg_price={row['Avg Price']}, capacity_coefficient={row['Flex Cpy Coef']}",
                                metadata={
                                    'OD': row['OD'],
                                    'time': row['Departure Time'],
                                    'product': row['Oneway_Product'],
                                    'avg_pax': row['Avg Pax'],
                                    'avg_price': row['Avg Price'],
                                    'capacity_coefficient': row['Flex Cpy Coef']
                                }
                            ))

                        embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
                        new_vectors = FAISS.from_documents(new_docs, embeddings)
                        return new_vectors
                    def retrieve_key_information(query):
                        matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                        od = []
                        t = []
                        for match in matches:
                            origin = match[0][2]
                            destination = match[0][7]
                            time = match[1]
                            od_pair = str((origin, destination))
                            od.append(od_pair)
                            t.append(time)
                        return od, t
                    def retrieve_time_period(departure_time):
                        intervals = {
                            '12pm~6pm': (time_class(12, 0), time_class(18, 0)),
                            '6pm~10pm': (time_class(18, 0), time_class(22, 0)),
                            '10pm~8am': (time_class(22, 0), time_class(8, 0)),
                            '8am~12pm': (time_class(8, 0), time_class(12, 0))
                        }

                        if isinstance(departure_time, str):
                            try:
                                hours, minutes = map(int, departure_time.split(':'))
                                departure_time = time_class(hours, minutes)
                            except ValueError:
                                raise ValueError("Time format should be: 'HH:MM'")

                        for interval_name, (start, end) in intervals.items():
                            if start < end:
                                if start <= departure_time < end:
                                    return interval_name
                            else:
                                if departure_time >= start or departure_time < end:
                                    return interval_name

                        return "Unknown"
                    def retrieve_parameter(origin,time_interval,product):
                        v1,v2,df = LoadFiles()
                        time_interval = f'({time_interval})'
                        key = product + '*' + time_interval
                        _value_ = 0
                        _ratio_ = 0
                        no_purchase_value = 0
                        no_purchase_value_ratio = 0
                        subset = v1[v1['Origin'] == origin]
                        if key in subset.columns and not subset.empty:
                            _value_ = subset[key].values[0]
                        subset2 = v2[v2['Origin'] == origin]
                        if key in subset2.columns and not subset2.empty:
                            _ratio_ = subset2[key].values[0]

                        if 'no_purchase' in subset.columns and not subset.empty:
                            no_purchase_value = subset['no_purchase'].values[0]
                        if 'no_purchase' in subset2.columns and not subset2.empty:
                            no_purchase_value_ratio = subset2['no_purchase'].values[0]
                        return _value_,_ratio_,no_purchase_value,no_purchase_value_ratio

                    def generate_coefficients(origin,time):
                        value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                        value_0_list, ratio_0_list = [], []

                        departure_time = datetime.strptime(time, '%H:%M').time()
                        time_interval = retrieve_time_period(departure_time)
                        value_1,ratio_1,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_flexi')

                        value_2,ratio_2,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_lite')

                        return value_1,ratio_1,value_2,ratio_2,value_0,ratio_0


                    def clean_text_preserve_newlines(text):
                        cleaned = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
                        cleaned = re.sub(r'[^\x20-\x7E\n]', '', cleaned)
                        cleaned = re.sub(r'(\n\s+)(\w+\s*=)', r'\n\2', cleaned)
                        cleaned = re.sub(r'\[\s+', '[', cleaned)
                        cleaned = re.sub(r'\s+\]', ']', cleaned)
                        cleaned = re.sub(r',\s+', ', ', cleaned)

                        return cleaned

                    # def PreProcessQuery(query):
                    def csv_qa_tool_flow(query: str):
                        new_vectors = New_Vectors(query)
                        matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                        num_match = re.search(r"optimal (\d+) flights", query)
                        num_flights = int(num_match.group(1)) if num_match else None  # 3
                        capacity_match = re.search(r"each Eco_flex ticket consumes (\d+) units of flight capacity", query)

                        if capacity_match:
                            eco_flex_capacity = capacity_match.group(1)
                        else:
                            eco_flex_capacity = 1.2
                        sigma_inflow_A = []
                        sigma_outflow_A = []
                        sigma_inflow_B = []
                        sigma_outflow_B = []
                        sigma_inflow_C = []
                        sigma_outflow_C = []
                        a_origin_flights_A_out = [
                            (od, time)
                            for (od, time) in matches
                            if od[2] == 'A'
                        ]

                        a_origin_flights_B_out = [
                            (od, time)
                            for (od, time) in matches
                            if od[2] == 'B'
                        ]

                        a_origin_flights_C_out = [
                            (od, time)
                            for (od, time) in matches
                            if od[2] == 'C'
                        ]

                        a_origin_flights_A = [
                            (od, time)
                            for (od, time) in matches
                            if od[7] == 'A'
                        ]

                        a_origin_flights_B = [
                            (od, time)
                            for (od, time) in matches
                            if od[7] == 'B'
                        ]

                        a_origin_flights_C = [
                            (od, time)
                            for (od, time) in matches
                            if od[7] == 'C'
                        ]

                        for od, time in a_origin_flights_A:
                            flight_name = od[2]+od[7]+time
                            sigma_inflow_A.append(flight_name)

                        for od, time in a_origin_flights_B:
                            flight_name = od[2]+od[7]+time
                            sigma_inflow_B.append(flight_name)

                        for od, time in a_origin_flights_C:
                            flight_name = od[2]+od[7]+time
                            sigma_inflow_C.append(flight_name)

                        for od, time in a_origin_flights_A_out:
                            flight_name = od[2]+od[7]+time
                            sigma_outflow_A.append(flight_name)

                        for od, time in a_origin_flights_B_out:
                            flight_name = od[2]+od[7]+time
                            sigma_outflow_B.append(flight_name)

                        for od, time in a_origin_flights_C_out:
                            flight_name = od[2]+od[7]+time
                            sigma_outflow_C.append(flight_name)

                        od_list_f, time_list_f, product_list_f, pax_list_f, price_list_f, capacitycoeff_list_f = [], [], [], [], [], []
                        od_list_l, time_list_l, product_list_l, pax_list_l, price_list_l, capacitycoeff_list_l = [], [], [], [], [], []
                        pt = []
                        value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                        value_0_list, ratio_0_list = [], []
                        for match in matches:
                            origin = match[0][2]
                            destination = match[0][7]
                            time = match[1]
                            od = str((origin, destination))
                            code = f"{origin}{destination}{time}"
                            pt.append(code)
                            retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                            doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_pax=")
                            for doc in doc_1:
                                content = doc.page_content
                                pattern = r',\s*(?=\w+=)'
                                parts = re.split(pattern, content)

                                pairs = [p.strip().replace('"', "'") for p in parts]
                                for pair in pairs:
                                    key, value = pair.split('=')
                                    if key == 'OD':
                                        od_list_f.append(value)
                                    elif key == 'Departure Time':
                                        time_list_f.append(value)
                                    elif key == 'Oneway_Product':
                                        product_list_f.append(value)
                                    elif key == 'avg_pax':
                                        pax_list_f.append(value)
                                    elif key == 'avg_price':
                                        price_list_f.append(value)

                            doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_pax=")
                            for doc in doc_2:
                                content = doc.page_content
                                pattern = r',\s*(?=\w+=)'
                                parts = re.split(pattern, content)

                                pairs = [p.strip().replace('"', "'") for p in parts]
                                for pair in pairs:
                                    key, value = pair.split('=')
                                    if key == 'OD':
                                        od_list_l.append(value)
                                    elif key == 'Departure Time':
                                        time_list_l.append(value)
                                    elif key == 'Oneway_Product':
                                        product_list_l.append(value)
                                    elif key == 'avg_pax':
                                        pax_list_l.append(value)
                                    elif key == 'avg_price':
                                        price_list_l.append(value)


                            value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(origin,time)
                            value_f_list.append(str(value_1))
                            ratio_f_list.append(str(ratio_1))
                            value_0_list.append(str(value_0))
                            ratio_0_list.append(str(ratio_0))
                            value_l_list.append(str(value_2))
                            ratio_l_list.append(str(ratio_2))

                        doc =  f'pt = {pt} \n'
                        doc +=  f'sigma_inflow_A = {sigma_inflow_A} \n sigma_outflow_A = {sigma_outflow_A}\n sigma_inflow_B = {sigma_inflow_B}\n sigma_outflow_B = {sigma_outflow_B}\n sigma_inflow_C = {sigma_inflow_C}\n sigma_outflow_C = {sigma_outflow_C} \n'

                        doc += f"\n avg_pax_f={pax_list_f} \n avg_pax_l={pax_list_l} \n avg_price_f={price_list_f}  \n avg_price_l={price_list_l} \n value_f_list ={value_f_list}\n  ratio_f_list={ratio_f_list}\n  value_l_list={value_l_list}\n  ratio_l_list={ratio_l_list}\n  value_0_list={value_0_list}\n  ratio_0_list={ratio_0_list}"
                        doc += f"\n option_num = {num_flights} \n"
                        doc += f"capacity_consum = {eco_flex_capacity}"

                        return doc


                    def FlowAgent():
                        problem_description = '''

                    Based on flight ticket options provided in './Test_Dataset/Air_NRM/information.csv', along with their associated attraction values (v1) and shadow attraction value ratios (v2), develop a Sales-Based Linear Programming (SBLP) model. The goal of this model is to recommend the optimal 3 flights that maximize total ticket sale revenue, specifically among flights with an origin-destination: OD = ('A', 'B') and a departure period (11am-1pm) in which the flights are: [(OD = ('A', 'B') AND Departure Time='11:20'), (OD = ('A', 'B') AND Departure Time='12:40')]

                    '''
                        example_data_description = csv_qa_tool_flow(problem_description)
                        example_matches = retrieve_key_information(problem_description)
                        fewshot_example = f'''
                        Question: {problem_description}

                        Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. The given OD and Departure Time values are {example_matches}

                        Action: CSVQA

                        Action Input: {problem_description}

                        Observation: {example_data_description}

                        Thought: Now I have known the answer.

                        Final Answer:

                        max \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                        capacity_consum*x_f[i] + x_l[i] <= 187
                        ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                        x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                        x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                        x_f[i] <= 10000* y[i]
                        x_l[i] <= 10000* y[i]
                        x_o[i] <= 10000* y[i]
                        \sum_i \in sigma_inflow_A y[i] = \sum_i \in sigma_outflow_A y[i]
                        \sum_i \in sigma_inflow_B y[i] = \sum_i \in sigma_outflow_B y[i]
                        \sum_i \in sigma_inflow_C y[i] = \sum_i \in sigma_outflow_C y[i]
                        \sum_i y[i] <= option_num
                        x_f[i],x_l[i],x_o[i] >= 0
                        y[i] is binary, where decision variables are based on the list pt and x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]. To be more specific, {example_data_description}
                        '''

                        tools = [Tool(name="CSVQA", func=csv_qa_tool_flow, description="Retrieve flight data.")]

                        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=user_api_key)
                        prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

                                Please refer to the following example and generate the answer in the same format:

                                {fewshot_example}

                                Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

                                When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

                                """

                        suffix = """

                                Begin!

                                User Description: {input}
                                {agent_scratchpad}"""


                        agent2 = initialize_agent(
                            tools,
                            llm=llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            agent_kwargs={
                                "prefix": prefix,
                                "suffix": suffix,
                            },
                            verbose=True,
                            handle_parsing_errors=True
                        )
                        return agent2

                    def policy_sblp_flow_model_code(query):
                        agent2 = FlowAgent()
                        llm_code = ChatOpenAI(
                                    temperature=0.0, model_name="gpt-4", openai_api_key=user_api_key
                                )
                        old_stdout = sys.stdout
                        sys.stdout = buffer = StringIO()
                        result = agent2.invoke({"input": query})
                        output_model = result['output']
                        sys.stdout = old_stdout
                        verbose_logs = buffer.getvalue()
                        observations = re.findall(r"Observation: (.*?)(?=\nThought:|\nFinal Answer:)", verbose_logs, re.DOTALL)
                        if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_model:
                                format = '''
                                max \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                                capacity_consum*x_f[i] + x_l[i] <= 187
                                ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                                x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                                x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                                x_f[i] <= 10000* y[i]
                                x_l[i] <= 10000* y[i]
                                x_o[i] <= 10000* y[i]
                                \sum_i \in sigma_inflow_A y[i] = \sum_i \in sigma_outflow_A y[i]
                                \sum_i \in sigma_inflow_B y[i] = \sum_i \in sigma_outflow_B y[i]
                                \sum_i \in sigma_inflow_C y[i] = \sum_i \in sigma_outflow_C y[i]
                                \sum_i y[i] <= option_num
                                x_f[i],x_l[i],x_o[i] >= 0
                                y[i] is binary, where decision variables are based on the list pt and x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]. To be more specific,
                        '''
                                text = re.sub(r'\[\d+m', '', str(observations[0]))
                                output_model = format + text

                        prompt = f"""
                        You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

                        Mathematical Optimization Model:
                        {output_model}

                        For example, here is a simple instance for reference:

                        Mathematical Optimization Model:

                        Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                        Capacity Constraints:

                        capacity_consum*x_f[i] + x_l[i] <= 187

                        Balance Constraints:

                        ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                        Scale Constraints:
                        x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                        x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                        M Constraints:

                        x_f[i] <= 10000* y[i]
                        x_l[i] <= 10000* y[i]
                        x_o[i] <= 10000* y[i]

                        Flow Conservation Constraints:

                        \sum_i \in sigma_inflow_A y[i] = \sum_i \in sigma_outflow_A y[i]

                        \sum_i \in sigma_inflow_B y[i] = \sum_i \in sigma_outflow_B y[i]

                        \sum_i \in sigma_inflow_C y[i] = \sum_i \in sigma_outflow_C y[i]

                        Cardinality Constraints:

                        \sum_i y[i] <= option_num

                        Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                        Binary Constraints: y[i] is binary, where decision variables are based on the list pt = ['CB10:45', 'AB11:20', 'BC11:20', 'BA12:25', 'BC12:25', 'CB12:35', 'AB12:40', 'AB12:55', 'AC13:05', 'AB13:55', 'CB14:15', 'CA14:15', 'BA14:25', 'BC14:25', 'AC15:45', 'AB16:05', 'BC16:30', 'CB16:55', 'CA16:55', 'AB17:05', 'AC17:25', 'AB18:00', 'CB18:30', 'CA18:30'], x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]
                        sigma_inflow_A = ['BA12:25', 'CA14:15', 'BA14:25', 'CA16:55', 'CA18:30'], sigma_outflow_A = ['AB11:20', 'AB12:40', 'AB12:55', 'AC13:05', 'AB13:55', 'AC15:45', 'AB16:05', 'AB17:05', 'AC17:25', 'AB18:00'], sigma_inflow_B = ['CB10:45', 'AB11:20', 'CB12:35', 'AB12:40', 'AB12:55', 'AB13:55', 'CB14:15', 'AB16:05', 'CB16:55', 'AB17:05', 'AB18:00', 'CB18:30'], sigma_outflow_B = ['BC11:20', 'BA12:25', 'BC12:25', 'BA14:25', 'BC14:25', 'BC16:30'], sigma_inflow_C = ['BC11:20', 'BC12:25', 'AC13:05', 'BC14:25', 'AC15:45', 'BC16:30', 'AC17:25'], sigma_outflow_C = ['CB10:45', 'CB12:35', 'CB14:15', 'CA14:15', 'CB16:55', 'CA16:55', 'CB18:30', 'CA18:30']

                        avg_pax_f=['127.86', '160.71', '133.14', '103.79', '36.0', '156.57', '87.36', '131.43', '111.64', '142.43', '33.71', '91.64', '133.0', '9.4', '133.71', '167.79', '92.36', '2.0', '149.29', '152.36', '130.14', '104.14', '13.57', '149.07']
                        avg_pax_l=['15.5', '18.64', '14.5', '23.36', '3.79', '19.93', '25.0', '20.29', '23.79', '28.21', '5.93', '17.86', '28.93', '1.67', '20.5', '25.43', '6.36', '2.4', '24.5', '27.14', '19.14', '13.64', '3.38', '17.43']
                        avg_price_f=['886.96', '1464.07', '873.59', '1479.64', '1520.65', '883.77', '1467.29', '856.66', '1663.16', '1477.78', '1510.9', '1638.05', '1480.24', '1521.05', '1669.65', '852.25', '907.06', '1583.0', '1665.33', '1475.07', '1670.39', '833.23', '1523.0', '1665.21']
                        avg_price_l=['263.88', '441.88', '269.24', '470.28', '596.44', '299.39', '505.8', '279.25', '565.96', '464.67', '496.51', '531.5', '470.65', '821.73', '506.95', '275.1', '268.12', '558.48', '493.28', '461.18', '473.92', '261.39', '443.91', '463.64']
                        value_f_list =['1.916', '2.05', '1.864', '3.079', '3.079', '2.826', '2.803', '2.803', '2.803', '2.803', '2.826', '2.826', '3.079', '3.079', '2.803', '2.803', '3.079', '2.826', '2.826', '2.803', '2.803', '1.624', '3.126', '3.126']
                        ratio_f_list=['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72', '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72', '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']
                        value_l_list=['1', '1', '1', '1.652', '1.652', '1.475', '1.367', '1.367', '1.367', '1.367', '1.475', '1.475', '1.652', '1.652', '1.367', '1.367', '1.652', '1.475', '1.475', '1.367', '1.367', '0.793', '1.631', '1.631']
                        ratio_l_list=['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72', '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72', '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']
                        value_0_list=['1.2', '0.9', '2.0', '2.0', '2.0', '1.2', '0.9', '0.9', '0.9', '0.9', '1.2', '1.2', '2.0', '2.0', '0.9', '0.9', '2.0', '1.2', '1.2', '0.9', '0.9', '0.9', '1.2', '1.2']
                        ratio_0_list=['1.72', '4.92', '3.7', '3.7', '3.7', '1.72', '4.92', '4.92', '4.92', '4.92', '1.72', '1.72', '3.7', '3.7', '4.92', '4.92', '3.7', '1.72', '1.72', '4.92', '4.92', '4.92', '1.72', '1.72']
                        option_num = 4

                        capacity_consum = 3

                        The corresponding Python code for this instance is as follows:

                        ```python
                        import gurobipy as gp
                        from gurobipy import GRB

                        # Create model
                        model = gp.Model("FlightTicketOptimization")

                        # Data
                        pt = ['CB10:45', 'AB11:20', 'BC11:20', 'BA12:25', 'BC12:25', 'CB12:35', 'AB12:40',
                            'AB12:55', 'AC13:05', 'AB13:55', 'CB14:15', 'CA14:15', 'BA14:25', 'BC14:25',
                            'AC15:45', 'AB16:05', 'BC16:30', 'CB16:55', 'CA16:55', 'AB17:05', 'AC17:25',
                            'AB18:00', 'CB18:30', 'CA18:30']

                        # Convert all string numbers to float
                        avg_pax_f = [float(x) for x in ['127.86', '160.71', '133.14', '103.79', '36.0', '156.57', '87.36',
                                                        '131.43', '111.64', '142.43', '33.71', '91.64', '133.0', '9.4',
                                                        '133.71', '167.79', '92.36', '2.0', '149.29', '152.36', '130.14',
                                                        '104.14', '13.57', '149.07']]

                        avg_pax_l = [float(x) for x in ['15.5', '18.64', '14.5', '23.36', '3.79', '19.93', '25.0',
                                                        '20.29', '23.79', '28.21', '5.93', '17.86', '28.93', '1.67',
                                                        '20.5', '25.43', '6.36', '2.4', '24.5', '27.14', '19.14',
                                                        '13.64', '3.38', '17.43']]

                        avg_price_f = [float(x) for x in ['886.96', '1464.07', '873.59', '1479.64', '1520.65', '883.77',
                                                        '1467.29', '856.66', '1663.16', '1477.78', '1510.9', '1638.05',
                                                        '1480.24', '1521.05', '1669.65', '852.25', '907.06', '1583.0',
                                                        '1665.33', '1475.07', '1670.39', '833.23', '1523.0', '1665.21']]

                        avg_price_l = [float(x) for x in ['263.88', '441.88', '269.24', '470.28', '596.44', '299.39',
                                                        '505.8', '279.25', '565.96', '464.67', '496.51', '531.5',
                                                        '470.65', '821.73', '506.95', '275.1', '268.12', '558.48',
                                                        '493.28', '461.18', '473.92', '261.39', '443.91', '463.64']]

                        value_f_list = [float(x) for x in ['1.916', '2.05', '1.864', '3.079', '3.079', '2.826', '2.803',
                                                        '2.803', '2.803', '2.803', '2.826', '2.826', '3.079', '3.079',
                                                        '2.803', '2.803', '3.079', '2.826', '2.826', '2.803', '2.803',
                                                        '1.624', '3.126', '3.126']]

                        ratio_f_list = [float(x) for x in ['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72',
                                                        '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72',
                                                        '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']]

                        value_l_list = [float(x) for x in ['1', '1', '1', '1.652', '1.652', '1.475', '1.367', '1.367',
                                                        '1.367', '1.367', '1.475', '1.475', '1.652', '1.652', '1.367',
                                                        '1.367', '1.652', '1.475', '1.475', '1.367', '1.367', '0.793',
                                                        '1.631', '1.631']]

                        ratio_l_list = [float(x) for x in ['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72',
                                                        '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72',
                                                        '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']]

                        value_0_list = [float(x) for x in ['1.2', '0.9', '2.0', '2.0', '2.0', '1.2', '0.9', '0.9', '0.9',
                                                        '0.9', '1.2', '1.2', '2.0', '2.0', '0.9', '0.9', '2.0', '1.2',
                                                        '1.2', '0.9', '0.9', '0.9', '1.2', '1.2']]

                        ratio_0_list = [float(x) for x in ['1.72', '4.92', '3.7', '3.7', '3.7', '1.72', '4.92', '4.92',
                                                        '4.92', '4.92', '1.72', '1.72', '3.7', '3.7', '4.92', '4.92',
                                                        '3.7', '1.72', '1.72', '4.92', '4.92', '4.92', '1.72', '1.72']]

                        # Flow conservation sets
                        sigma_inflow_A = ['BA12:25', 'CA14:15', 'BA14:25', 'CA16:55', 'CA18:30']
                        sigma_outflow_A = ['AB11:20', 'AB12:40', 'AB12:55', 'AC13:05', 'AB13:55', 'AC15:45',
                                        'AB16:05', 'AB17:05', 'AC17:25', 'AB18:00']
                        sigma_inflow_B = ['CB10:45', 'AB11:20', 'CB12:35', 'AB12:40', 'AB12:55', 'AB13:55',
                                        'CB14:15', 'AB16:05', 'CB16:55', 'AB17:05', 'AB18:00', 'CB18:30']
                        sigma_outflow_B = ['BC11:20', 'BA12:25', 'BC12:25', 'BA14:25', 'BC14:25', 'BC16:30']
                        sigma_inflow_C = ['BC11:20', 'BC12:25', 'AC13:05', 'BC14:25', 'AC15:45', 'BC16:30', 'AC17:25']
                        sigma_outflow_C = ['CB10:45', 'CB12:35', 'CB14:15', 'CA14:15', 'CB16:55', 'CA16:55',
                                        'CB18:30', 'CA18:30']

                        option_num = 4
                        capacity_consum = 3

                        # Decision variables
                        x_f = model.addVars(pt, name="x_f")  # Flexible tickets
                        x_l = model.addVars(pt, name="x_l")  # Limited tickets
                        x_o = model.addVars(pt, name="x_o")  # Other tickets
                        y = model.addVars(pt, vtype=GRB.BINARY, name="y")  # Flight selection

                        # Objective: Maximize revenue
                        model.setObjective(
                            gp.quicksum(avg_price_f[i] * x_f[pt[i]] + avg_price_l[i] * x_l[pt[i]] for i in range(len(pt))),
                            GRB.MAXIMIZE
                        )

                        # Capacity constraints
                        for i in range(len(pt)):
                            model.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187, f"capacity_{{pt[i]}}")

                        # Balance constraints
                        for i in range(len(pt)):
                            model.addConstr(
                                ratio_f_list[i] * x_f[pt[i]] + ratio_l_list[i] * x_l[pt[i]] + ratio_0_list[i] * x_o[pt[i]]
                                <= avg_pax_f[i] + avg_pax_l[i],
                                f"balance_{{pt[i]}}"
                            )

                        # Scale constraints
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]]/value_f_list[i] - x_o[pt[i]]/value_0_list[i] <= 0, f"scale_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]]/value_l_list[i] - x_o[pt[i]]/value_0_list[i] <= 0, f"scale_l_{{pt[i]}}")

                        # M constraints (linking constraints)
                        M = 10000
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]] <= M * y[pt[i]], f"M_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]] <= M * y[pt[i]], f"M_l_{{pt[i]}}")
                            model.addConstr(x_o[pt[i]] <= M * y[pt[i]], f"M_o_{{pt[i]}}")

                        # Flow conservation constraints
                        model.addConstr(
                            gp.quicksum(y[flight] for flight in sigma_inflow_A) ==
                            gp.quicksum(y[flight] for flight in sigma_outflow_A),
                            "flow_conservation_A"
                        )

                        model.addConstr(
                            gp.quicksum(y[flight] for flight in sigma_inflow_B) ==
                            gp.quicksum(y[flight] for flight in sigma_outflow_B),
                            "flow_conservation_B"
                        )

                        model.addConstr(
                            gp.quicksum(y[flight] for flight in sigma_inflow_C) ==
                            gp.quicksum(y[flight] for flight in sigma_outflow_C),
                            "flow_conservation_C"
                        )

                        # Cardinality constraint
                        model.addConstr(gp.quicksum(y[pt[i]] for i in range(len(pt))) <= option_num, "cardinality")

                        # Non-negativity constraints
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]] >= 0, f"nonneg_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]] >= 0, f"nonneg_l_{{pt[i]}}")
                            model.addConstr(x_o[pt[i]] >= 0, f"nonneg_o_{{pt[i]}}")

                        # Optimize model
                        model.optimize()
                        if model.status == GRB.OPTIMAL:
                            print("Optimal Solution Found:")
                            # Print decision variable values
                            for var in model.getVars():
                                if 'y' in var.VarName and var.X>0:
                                    print(var.VarName,var.X)
                        else:
                            print("No optimal solution found. Status Code:", m.status)

                        """

                        messages = [
                            HumanMessage(content=prompt)
                        ]

                        response = llm_code(messages)
                        output_code = response.content
                        if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_code:
                            format_code = '''
                        import gurobipy as gp
                        from gurobipy import GRB

                        # Create model
                        model = gp.Model("FlightTicketOptimization")
                        x_f = model.addVars(pt, name="x_f")  # Flexible tickets
                        x_l = model.addVars(pt, name="x_l")  # Limited tickets
                        x_o = model.addVars(pt, name="x_o")  # Other tickets
                        y = model.addVars(pt, vtype=GRB.BINARY, name="y")  # Flight selection

                        # Objective: Maximize revenue
                        model.setObjective(
                            gp.quicksum(avg_price_f[i] * x_f[pt[i]] + avg_price_l[i] * x_l[pt[i]] for i in range(len(pt))),
                            GRB.MAXIMIZE
                        )

                        # Capacity constraints
                        for i in range(len(pt)):
                            model.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187, f"capacity_{{pt[i]}}")

                        # Balance constraints
                        for i in range(len(pt)):
                            model.addConstr(
                                float(ratio_f_list[i]) * x_f[pt[i]] + float(ratio_l_list[i]) * x_l[pt[i]] + float(ratio_0_list[i]) * x_o[pt[i]]
                                <= float(avg_pax_f[i]) + float(avg_pax_l[i]),
                                f"balance_{{pt[i]}}"
                            )

                        # Scale constraints
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]]/float(value_f_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]]/float(value_l_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_l_{{pt[i]}}")

                        # M constraints (linking constraints)
                        M = 10000
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]] <= M * y[pt[i]], f"M_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]] <= M * y[pt[i]], f"M_l_{{pt[i]}}")
                            model.addConstr(x_o[pt[i]] <= M * y[pt[i]], f"M_o_{{pt[i]}}")

                        # Flow conservation constraints
                        model.addConstr(
                            gp.quicksum(y[flight] for flight in sigma_inflow_A) ==
                            gp.quicksum(y[flight] for flight in sigma_outflow_A),
                            "flow_conservation_A"
                        )

                        model.addConstr(
                            gp.quicksum(y[flight] for flight in sigma_inflow_B) ==
                            gp.quicksum(y[flight] for flight in sigma_outflow_B),
                            "flow_conservation_B"
                        )

                        model.addConstr(
                            gp.quicksum(y[flight] for flight in sigma_inflow_C) ==
                            gp.quicksum(y[flight] for flight in sigma_outflow_C),
                            "flow_conservation_C"
                        )

                        # Cardinality constraint
                        model.addConstr(gp.quicksum(y[pt[i]] for i in range(len(pt))) <= option_num, "cardinality")

                        # Non-negativity constraints
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]] >= 0, f"nonneg_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]] >= 0, f"nonneg_l_{{pt[i]}}")
                            model.addConstr(x_o[pt[i]] >= 0, f"nonneg_o_{{pt[i]}}")

                        # Optimize model
                        model.optimize()

                        # Save model to file for inspection
                        model.write("flight_optimization.lp")
                        if model.status == GRB.OPTIMAL:
                            print("Optimal Solution Found:")
                            # Print decision variable values
                            for var in model.getVars():
                                if 'y' in var.VarName and var.X>0:
                                    print(var.VarName,var.X)
                        else:
                            print("No optimal solution found. Status Code:", m.status)
                        '''
                            text = clean_text_preserve_newlines(str(observations[0]))
                            output_code = text + format_code
                        return output_model,output_code


                    def ProcessPolicyFlow(query):
                        output_model, response = policy_sblp_flow_model_code(query)
                        return output_model, response

                    def csv_qa_tool_no_flow(query: str):
                        new_vectors = New_Vectors(query)
                        matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                        num_match = re.search(r"optimal (\d+) flights", query)
                        num_flights = int(num_match.group(1)) if num_match else None  # 3
                        capacity_match = re.search(r"each Eco_flex ticket consumes (\d+) units of flight capacity", query)
                        if capacity_match:
                            eco_flex_capacity = capacity_match.group(1)
                        else:
                            eco_flex_capacity = 1.2
                        od_list_f, time_list_f, product_list_f, pax_list_f, price_list_f, capacitycoeff_list_f = [], [], [], [], [], []
                        od_list_l, time_list_l, product_list_l, pax_list_l, price_list_l, capacitycoeff_list_l = [], [], [], [], [], []
                        pt = []
                        value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                        value_0_list, ratio_0_list = [], []
                        for match in matches:
                            origin = match[0][2]
                            destination = match[0][7]
                            time = match[1]
                            od = str((origin, destination))
                            code = f"{origin}{destination}{time}"
                            pt.append(code)
                            retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                            doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_pax=")
                            for doc in doc_1:
                                content = doc.page_content
                                pattern = r',\s*(?=\w+=)'
                                parts = re.split(pattern, content)

                                pairs = [p.strip().replace('"', "'") for p in parts]
                                for pair in pairs:
                                    key, value = pair.split('=')
                                    if key == 'OD':
                                        od_list_f.append(value)
                                    elif key == 'Departure Time':
                                        time_list_f.append(value)
                                    elif key == 'Oneway_Product':
                                        product_list_f.append(value)
                                    elif key == 'avg_pax':
                                        pax_list_f.append(value)
                                    elif key == 'avg_price':
                                        price_list_f.append(value)
                                    elif key == 'capacity_coefficient':
                                        capacitycoeff_list_f.append(value)

                            doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_pax=")
                            for doc in doc_2:
                                content = doc.page_content
                                pattern = r',\s*(?=\w+=)'
                                parts = re.split(pattern, content)

                                pairs = [p.strip().replace('"', "'") for p in parts]
                                for pair in pairs:
                                    key, value = pair.split('=')
                                    if key == 'OD':
                                        od_list_l.append(value)
                                    elif key == 'Departure Time':
                                        time_list_l.append(value)
                                    elif key == 'Oneway_Product':
                                        product_list_l.append(value)
                                    elif key == 'avg_pax':
                                        pax_list_l.append(value)
                                    elif key == 'avg_price':
                                        price_list_l.append(value)
                                    elif key == 'capacity_coefficient':
                                        capacitycoeff_list_l.append(value)

                            value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(origin,time)
                            value_f_list.append(str(value_1))
                            ratio_f_list.append(str(ratio_1))
                            value_0_list.append(str(value_0))
                            ratio_0_list.append(str(ratio_0))
                            value_l_list.append(str(value_2))
                            ratio_l_list.append(str(ratio_2))

                        doc =  f'decision variables are based on the list pt = {pt}, x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt] \n'
                        doc += f"\n avg_pax_f={pax_list_f} \n avg_pax_l={pax_list_l} \n avg_price_f={price_list_f}  \n avg_price_l={price_list_l}  \n value_f_list ={value_f_list}\n  ratio_f_list={ratio_f_list}\n  value_l_list={value_l_list}\n  ratio_l_list={ratio_l_list}\n  value_0_list={value_0_list}\n  ratio_0_list={ratio_0_list}"
                        doc += f"\n option_num = {num_flights} \n"
                        doc += f"\n capacity_consum = {eco_flex_capacity} \n"
                        return doc

                    def NoFlowAgent():
                        problem_description = '''

                    Based on flight ticket options provided in './Test_Dataset/Air_NRM/information.csv', along with their associated attraction values (v1) and shadow attraction value ratios (v2), develop a Sales-Based Linear Programming (SBLP) model. The goal of this model is to recommend the optimal 3 flights that maximize total ticket sale revenue, specifically among flights with an origin-destination: OD = ('A', 'B') and a departure period (11am-1pm) in which the flights are: [(OD = ('A', 'B') AND Departure Time='11:20'), (OD = ('A', 'B') AND Departure Time='12:40')]

                    '''
                        example_data_description = csv_qa_tool_no_flow(problem_description)
                        example_matches = retrieve_key_information(problem_description)
                        fewshot_example = f'''
                    Question: {problem_description}

                    Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. The given OD and Departure Time values are {example_matches}

                    Action: CSVQA

                    Action Input: {problem_description}

                    Observation: {example_data_description}

                    Thought: Now I have known the answer.

                    Final Answer:

                    Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                    Capacity Constraints:

                    capacity_consum*x_f[i] + x_l[i] <= 187

                    Balance Constraints:

                    ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                    Scale Constraints:
                    x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                    x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                    M Constraints:

                    x_f[i] <= 10000* y[i]
                    x_l[i] <= 10000* y[i]
                    x_o[i] <= 10000* y[i]

                    Cardinality Constraints:

                    \sum_i y[i] <= option_num

                    Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                    Binary Constraints: y[i] is binary, where {example_data_description}
                    '''

                        tools = [Tool(name="CSVQA", func=csv_qa_tool_no_flow, description="Retrieve flight data.")]

                        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=user_api_key)
                        prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

                                Please refer to the following example and generate the answer in the same format:

                                {fewshot_example}

                                Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

                                When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

                                """

                        suffix = """

                            Begin!

                            User Description: {input}
                            {agent_scratchpad}"""



                        format = '''

                    Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                    Capacity Constraints:

                    capacity_consum*x_f[i] + x_l[i] <= 187

                    Balance Constraints:

                    ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                    Scale Constraints:
                    x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                    x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                    M Constraints:

                    x_f[i] <= 10000* y[i]
                    x_l[i] <= 10000* y[i]
                    x_o[i] <= 10000* y[i]

                    Cardinality Constraints:

                    \sum_i y[i] <= option_num

                    Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                    Binary Constraints: y[i] \in {0,1}
                    '''

                        agent2 = initialize_agent(
                        tools,
                        llm=llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        agent_kwargs={
                            "prefix": prefix,
                            "suffix": suffix,
                        },
                        verbose=True,
                        handle_parsing_errors=True
                        )

                        return agent2


                    def policy_sblp_noflow_model_code(query):
                        agent2 = NoFlowAgent()
                        llm_code = ChatOpenAI(
                                    temperature=0.0, model_name="gpt-4", openai_api_key=user_api_key
                                )
                        old_stdout = sys.stdout
                        sys.stdout = buffer = StringIO()
                        result = agent2.invoke({"input": query})
                        output_model = result['output']
                        sys.stdout = old_stdout
                        verbose_logs = buffer.getvalue()
                        observations = re.findall(r"Observation: (.*?)(?=\nThought:|\nFinal Answer:)", verbose_logs, re.DOTALL)
                        if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_model:
                            format = '''
                                max \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                                capacity_consum*x_f[i] + x_l[i] <= 187
                                ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                                x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                                x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                                x_f[i] <= 10000* y[i]
                                x_l[i] <= 10000* y[i]
                                x_o[i] <= 10000* y[i]
                                \sum_i y[i] <= option_num
                                x_f[i],x_l[i],x_o[i] >= 0
                                y[i] is binary, where decision variables are based on the list pt and x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]. To be more specific,
                            '''
                            text = re.sub(r'\[\d+m', '', str(observations[0]))
                            output_model = format + text


                        prompt = f"""
                        You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

                        Mathematical Optimization Model:
                        {output_model}

                        For example, here is a simple instance for reference:

                        Mathematical Optimization Model:

                        Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                        Capacity Constraints:

                        capacity_consum*x_f[i] + x_l[i] <= 187

                        Balance Constraints:

                        ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                        Scale Constraints:
                        x_f[i]/ratio_f_list[i] - x_o[i]/value_0_list[i] <=0
                        x_l[i]/ratio_l_list[i] - x_o[i]/value_0_list[i] <=0

                        M Constraints:

                        x_f[i] <= 10000* y[i]
                        x_l[i] <= 10000* y[i]
                        x_o[i] <= 10000* y[i]

                        Cardinality Constraints:

                        \sum_i y[i] <= option_num

                        Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                        Binary Constraints: y[i] \in {{0,1}}

                        , where decision variables are:
                        pt = ['BC12:25', 'BC14:25', 'BC15:40', 'BC09:05', 'BC11:20', 'BC16:30', 'BC19:05']
                        avg_pax_f=['36.0', '9.4', '11.69', '17.5', '133.14', '92.36', '137.14']
                        avg_pax_l=['3.79', '1.67', '3.27', '3.23', '14.5', '6.36', '17.57']
                        avg_price_f=['1520.65', '1521.05', '1544.13', '1489.26', '873.59', '907.06', '875.17']
                        avg_price_l=['596.44', '821.73', '483.3', '443.62', '269.24', '268.12', '282.71']
                        value_f_list =[np.float64(3.079), np.float64(3.079), np.float64(3.079), np.float64(1.864), np.float64(1.864), np.float64(3.079), np.float64(2.411)]
                        ratio_f_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                        value_l_list=[np.float64(1.652), np.float64(1.652), np.float64(1.652), np.int64(1), np.int64(1), np.float64(1.652), np.float64(1.293)]
                        ratio_l_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                        value_0_list=[np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0)]
                        ratio_0_list=[np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7)]
                        capacity_consum = 1.2


                        The corresponding Python code for this instance is as follows:

                        ```python
                        import gurobipy as gp
                        from gurobipy import GRB
                        import numpy as np

                        pt = ['BC12:25', 'BC14:25', 'BC15:40', 'BC09:05', 'BC11:20', 'BC16:30', 'BC19:05']
                        avg_pax_f=['36.0', '9.4', '11.69', '17.5', '133.14', '92.36', '137.14']
                        avg_pax_l=['3.79', '1.67', '3.27', '3.23', '14.5', '6.36', '17.57']
                        avg_price_f=['1520.65', '1521.05', '1544.13', '1489.26', '873.59', '907.06', '875.17']
                        avg_price_l=['596.44', '821.73', '483.3', '443.62', '269.24', '268.12', '282.71']
                        value_f_list =[np.float64(3.079), np.float64(3.079), np.float64(3.079), np.float64(1.864), np.float64(1.864), np.float64(3.079), np.float64(2.411)]
                        ratio_f_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                        value_l_list=[np.float64(1.652), np.float64(1.652), np.float64(1.652), np.int64(1), np.int64(1), np.float64(1.652), np.float64(1.293)]
                        ratio_l_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                        value_0_list=[np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0)]
                        ratio_0_list=[np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7)]
                        capacity_consum = 1.2

                        pax_f = []
                        for i in avg_pax_f:
                            i = float(i)
                            pax_f.append(i)

                        pax_l = []
                        for i in avg_pax_l:
                            pax_l.append(float(i))

                        price_f = []
                        for i in avg_price_f:
                            price_f.append(float(i))

                        price_l = []
                        for i in avg_price_l:
                            price_l.append(float(i))

                        coefficient_f = []

                        for i in capacity_coefficient_f:
                            coefficient_f.append(float(i))

                        coefficient_l = []

                        for i in capacity_coefficient_l:
                            coefficient_l.append(float(i))
                        # Create the model
                        m = gp.Model("Flight_Optimization")

                        # Decision variables
                        x_f = m.addVars(pt, vtype=GRB.CONTINUOUS, name="x_f")
                        x_l = m.addVars(pt, vtype=GRB.CONTINUOUS, name="x_l")
                        x_o = m.addVars(pt, vtype=GRB.CONTINUOUS, name="x_o")
                        y = m.addVars(pt, vtype=GRB.BINARY, name="y")

                        # Objective function
                        m.setObjective(
                            gp.quicksum(float(avg_price_f[i]) * x_f[pt[i]] + float(avg_price_l[i]) * x_l[pt[i]] for i in range(7)),
                            GRB.MAXIMIZE
                        )

                        # Constraints
                        for i in range(7):
                            m.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187)
                            m.addConstr(x_f[pt[i]] <= 10000 * y[pt[i]])
                            m.addConstr(x_l[pt[i]] <= 10000 * y[pt[i]])
                            m.addConstr(x_o[pt[i]] <= 10000 * y[pt[i]])

                            m.addConstr(ratio_f_list[i] * x_f[pt[i]] + ratio_l_list[i] * x_l[pt[i]] + ratio_0_list[i] * x_o[pt[i]] <= float(avg_pax_f[i]) + float(avg_pax_l[i]))

                            m.addConstr(x_f[pt[i]] / value_f_list[i] <= x_o[pt[i]] / value_0_list[i])
                            m.addConstr(x_l[pt[i]] / value_l_list[i] <= x_o[pt[i]] / value_0_list[i])


                        # Solve the model
                        m.optimize()
                        m.write('test.lp')

                        # Check if the model was solved successfully
                        if m.status == GRB.OPTIMAL:
                            print("Optimal Solution Found:")
                            # Print decision variable values
                            for var in m.getVars():
                                print(var.VarName,var.X)
                        else:
                            print("No optimal solution found. Status Code:", m.status)

                        """

                        messages = [
                            HumanMessage(content=prompt)
                        ]



                        response = llm_code(messages)
                        output_code = response.content

                        if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_code:
                            format_code = '''
                        import gurobipy as gp
                        from gurobipy import GRB

                        # Create model
                        model = gp.Model("FlightTicketOptimization")
                        x_f = model.addVars(pt, name="x_f")  # Flexible tickets
                        x_l = model.addVars(pt, name="x_l")  # Limited tickets
                        x_o = model.addVars(pt, name="x_o")  # Other tickets
                        y = model.addVars(pt, vtype=GRB.BINARY, name="y")  # Flight selection

                        # Objective: Maximize revenue
                        model.setObjective(
                            gp.quicksum(avg_price_f[i] * x_f[pt[i]] + avg_price_l[i] * x_l[pt[i]] for i in range(len(pt))),
                            GRB.MAXIMIZE
                        )

                        # Capacity constraints
                        for i in range(len(pt)):
                            model.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187, f"capacity_{{pt[i]}}")

                        # Balance constraints
                        for i in range(len(pt)):
                            model.addConstr(
                                float(ratio_f_list[i]) * x_f[pt[i]] + float(ratio_l_list[i]) * x_l[pt[i]] + float(ratio_0_list[i]) * x_o[pt[i]]
                                <= float(avg_pax_f[i]) + float(avg_pax_l[i]),
                                f"balance_{{pt[i]}}"
                            )

                        # Scale constraints
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]]/float(value_f_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]]/float(value_l_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_l_{{pt[i]}}")

                        # M constraints (linking constraints)
                        M = 10000
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]] <= M * y[pt[i]], f"M_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]] <= M * y[pt[i]], f"M_l_{{pt[i]}}")
                            model.addConstr(x_o[pt[i]] <= M * y[pt[i]], f"M_o_{{pt[i]}}")

                        # Cardinality constraint
                        model.addConstr(gp.quicksum(y[pt[i]] for i in range(len(pt))) <= option_num, "cardinality")

                        # Non-negativity constraints
                        for i in range(len(pt)):
                            model.addConstr(x_f[pt[i]] >= 0, f"nonneg_f_{{pt[i]}}")
                            model.addConstr(x_l[pt[i]] >= 0, f"nonneg_l_{{pt[i]}}")
                            model.addConstr(x_o[pt[i]] >= 0, f"nonneg_o_{{pt[i]}}")

                        # Optimize model
                        model.optimize()

                        # Save model to file for inspection
                        model.write("flight_optimization.lp")
                        if model.status == GRB.OPTIMAL:
                            print("Optimal Solution Found:")
                            # Print decision variable values
                            for var in model.getVars():
                                if 'y' in var.VarName and var.X>0:
                                    print(var.VarName,var.X)
                        else:
                            print("No optimal solution found. Status Code:", m.status)
                        '''
                        text = clean_text_preserve_newlines(str(observations[0]))
                        output_code = text + format_code

                        return output_model,output_code


                    def ProcessPolicyNoFlow(query):
                        output_model, response = policy_sblp_noflow_model_code(query)
                        return output_model, response
                    
                    def csv_qa_tool_CA(query: str):
                        new_vectors = New_Vectors(query)
                        matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                        num_match = re.search(r"optimal (\d+) flights", query)
                        num_flights = int(num_match.group(1)) if num_match else None
                        od_list_f, time_list_f, product_list_f, pax_list_f, price_list_f = [], [], [], [], []
                        od_list_l, time_list_l, product_list_l, pax_list_l, price_list_l = [], [], [], [], []
                        pt = []
                        value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                        value_0_list, ratio_0_list = [], []
                        for match in matches:
                            origin = match[0][2]
                            destination = match[0][7]
                            time = match[1]
                            od = str((origin, destination))
                            code = f"{origin}{destination}{time}"
                            pt.append(code)
                            retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                            doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_pax=")
                            for doc in doc_1:
                                content = doc.page_content
                                pattern = r',\s*(?=\w+=)'
                                parts = re.split(pattern, content)

                                pairs = [p.strip().replace('"', "'") for p in parts]
                                for pair in pairs:
                                    key, value = pair.split('=')
                                    if key == 'OD':
                                        od_list_f.append(value)
                                    elif key == 'Departure Time':
                                        time_list_f.append(value)
                                    elif key == 'Oneway_Product':
                                        product_list_f.append(value)
                                    elif key == 'avg_pax':
                                        pax_list_f.append(value)
                                    elif key == 'avg_price':
                                        price_list_f.append(value)


                            doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_pax=")
                            for doc in doc_2:
                                content = doc.page_content
                                pattern = r',\s*(?=\w+=)'
                                parts = re.split(pattern, content)

                                pairs = [p.strip().replace('"', "'") for p in parts]
                                for pair in pairs:
                                    key, value = pair.split('=')
                                    if key == 'OD':
                                        od_list_l.append(value)
                                    elif key == 'Departure Time':
                                        time_list_l.append(value)
                                    elif key == 'Oneway_Product':
                                        product_list_l.append(value)
                                    elif key == 'avg_pax':
                                        pax_list_l.append(value)
                                    elif key == 'avg_price':
                                        price_list_l.append(value)

                            value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(origin,time)
                            value_f_list.append(str(value_1))
                            ratio_f_list.append(str(ratio_1))
                            value_0_list.append(str(value_0))
                            ratio_0_list.append(str(ratio_0))
                            value_l_list.append(str(value_2))
                            ratio_l_list.append(str(ratio_2))

                        doc = f'Decision Variables are based on list I, I= {pt}'
                        doc += f"\n avg_pax_f={pax_list_f} \n avg_pax_l={pax_list_l} \n avg_price_f={price_list_f}  \n avg_price_l={price_list_l} \n value_f_list ={value_f_list}\n  ratio_f_list={ratio_f_list}\n  value_l_list={value_l_list}\n  ratio_l_list={ratio_l_list}\n  value_0_list={value_0_list}\n  ratio_0_list={ratio_0_list}"
                        return doc

                    def CA_Agent():
                        problem_description = '''

                    Based on flight ticket options provided in './Test_Dataset/Air_NRM/information.csv', along with their associated attraction values (v1) and shadow attraction value ratios (v2), develop a Sales-Based Linear Programming (SBLP) model. The goal of this model is to recommend the optimal 3 flights that maximize total ticket sale revenue, specifically among flights with an origin-destination: OD = ('A', 'B') and a departure period (11am-1pm) in which the flights are: [(OD = ('A', 'B') AND Departure Time='11:20'), (OD = ('A', 'B') AND Departure Time='12:40')]

                    '''
                        example_data_description = csv_qa_tool_CA(problem_description)
                        example_matches = retrieve_key_information(problem_description)
                        fewshot_example = f'''
                    Question: {problem_description}

                    Thought: I need to retrieve relevant information from './Test_Dataset/Air_NRM/information.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. The given OD and Departure Time values are {example_matches}

                    Action: CSVQA

                    Action Input: {problem_description}

                    Observation: {example_data_description}

                    Thought: Now I have known the answer.

                    Final Answer:

                    Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                    Capacity Constraints:

                    capacity_consum*x_f[i] + x_l[i] <= 187

                    Balance Constraints:

                    ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                    Scale Constraints:
                    x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                    x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                    M Constraints:

                    x_f[i] <= 10000* y[i]
                    x_l[i] <= 10000* y[i]
                    x_o[i] <= 10000* y[i]

                    Cardinality Constraints:

                    \sum_i y[i] <= option_num

                    Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                    Binary Constraints: y[i] is binary, where {example_data_description}
                    '''

                        tools = [Tool(name="CSVQA", func=csv_qa_tool_CA, description="Retrieve flight data.")]

                        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=user_api_key)
                        prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

                                Please refer to the following example and generate the answer in the same format:

                                {fewshot_example}

                                Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

                                When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

                                """

                        suffix = """

                            Begin!

                            User Description: {input}
                            {agent_scratchpad}"""



                        format = '''

                    Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                    Capacity Constraints:

                    capacity_consum*x_f[i] + x_l[i] <= 187

                    Balance Constraints:

                    ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                    Scale Constraints:
                    x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                    x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                    M Constraints:

                    x_f[i] <= 10000* y[i]
                    x_l[i] <= 10000* y[i]
                    x_o[i] <= 10000* y[i]

                    Cardinality Constraints:

                    \sum_i y[i] <= option_num

                    Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                    Binary Constraints: y[i] \in {0,1}
                    '''

                        agent2 = initialize_agent(
                        tools,
                        llm=llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        agent_kwargs={
                            "prefix": prefix,
                            "suffix": suffix,
                        },
                        verbose=True,
                        handle_parsing_errors=True
                        )

                        return agent2

                    def conversational_chat(query):
                        agent2 = CA_Agent()
                        result = agent2.invoke({"input": query})
                        output = result['output']
                        return output

                    def get_answer(query):
                        old_stdout = sys.stdout
                        sys.stdout = buffer = StringIO()
                        agent2 = CA_Agent()
                        result = agent2.invoke({"input": query})
                        output_model = result['output']
                        sys.stdout = old_stdout
                        verbose_logs = buffer.getvalue()
                        observations = re.findall(r"Observation: (.*?)(?=\nThought:|\nFinal Answer:)", verbose_logs, re.DOTALL)
                        if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" not in output_model:
                            format = '''
                            Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                            Capacity Constraints:
                            1.2*x_f[i] + x_l[i] <= 187
                            Balance Constraints:
                            ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                            Scale Constraints:
                            x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                            x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                            Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0,where
                            '''
                            text = re.sub(r'\[\d+m', '', str(observations[0]))
                            output_model = format + clean_text_preserve_newlines(text)

                        return output_model

                    def ProcessCA(query):
                        CA_result = get_answer(query)
                        return CA_result
                    
                    if "flow conservation constraints" in query:
                        # print("Recommend Optimal Flights With Flow Conervation Constraints")
                        output_model, code = ProcessPolicyFlow(query)
                        Type = "Policy_Flow"
                    elif "recommend the optimal" in query:
                        # print("Recommend Optimal Flights. No Flow Constraints.")
                        output_model, code = ProcessPolicyNoFlow(query)
                        Type = "Policy_NoFlow"
                    else:
                        # print("Only Develop Mathematic Formulations. No Recommendation for Flights.")
                        output_model = ProcessCA(query)
                        code = "This SBLP Problem Type is to develop mathematical model. Therefore, no code in this part."
                        Type = "CA"
                    
                    ai_response = f'It is a {Type} SBLP Problem,\n model: \n {output_model},\n code for this model: \n{code}'


                # if selected_problem == "Sales-Based Linear Programming":
                #     new_docs = []
                #     for _, row in df.iterrows():
                #         new_docs.append(Document(
                #             page_content=f"OD={row['OD']},Departure_Time_Flight1={row['Departure Time']},Oneway_Product={row['Oneway_Product']}, avg_pax={row['Avg Pax']}, avg_price={row['Avg Price']}, capacity_coefficient={row['Flex Cpy Coef']}",
                #             metadata={
                #                 'OD': row['OD'],
                #                 'time': row['Departure Time'],  
                #                 'product': row['Oneway_Product'],
                #                 'avg_pax': row['Avg Pax'],
                #                 'avg_price': row['Avg Price'],
                #                 'capacity_coefficient': row['Flex Cpy Coef']
                #             }
                #         ))

                #     embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
                #     new_vectors = FAISS.from_documents(new_docs, embeddings)
                    

                #     problem_description = '''

                #     Based on flight ticket options provided in 'information.csv', along with their associated attraction values (v1) and shadow attraction value ratios (v2), develop a Sales-Based Linear Programming (SBLP) model. The goal of this model is to recommend the optimal 3 flights that maximize total ticket sale revenue, specifically among flights with an origin-destination: OD = ('A', 'B') and a departure period (11am-1pm) in which the flights are: [(OD = ('A', 'B') AND Departure Time='11:20'), (OD = ('A', 'B') AND Departure Time='12:40')]

                #     '''
                #     v1 = pd.read_csv('./Test_Dataset/Air_NRM/v1.csv')
                #     v2 = pd.read_csv('./Test_Dataset/Air_NRM/v2.csv')
                #     df = pd.read_csv('./Test_Dataset/Air_NRM/information.csv')
                #     if "flow conservation constraints" in query:
                #         def csv_qa_tool(query: str):
                #             matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                #             num_match = re.search(r"optimal (\d+) flights", query)
                #             num_flights = int(num_match.group(1)) if num_match else None  # 3
                #             capacity_match = re.search(r"each Eco_flex ticket consumes (\d+) units of flight capacity", query)

                #             if capacity_match:
                #                 eco_flex_capacity = capacity_match.group(1)
                #             else:
                #                 eco_flex_capacity = 1.2
                #             sigma_inflow_A = []
                #             sigma_outflow_A = []
                #             sigma_inflow_B = []
                #             sigma_outflow_B = []
                #             sigma_inflow_C = []
                #             sigma_outflow_C = []
                #             a_origin_flights_A_out = [
                #                 (od, time)
                #                 for (od, time) in matches
                #                 if od[2] == 'A'
                #             ]

                #             a_origin_flights_B_out = [
                #                 (od, time)
                #                 for (od, time) in matches
                #                 if od[2] == 'B'
                #             ]

                #             a_origin_flights_C_out = [
                #                 (od, time)
                #                 for (od, time) in matches
                #                 if od[2] == 'C'
                #             ]

                #             a_origin_flights_A = [
                #                 (od, time)
                #                 for (od, time) in matches
                #                 if od[7] == 'A'
                #             ]

                #             a_origin_flights_B = [
                #                 (od, time)
                #                 for (od, time) in matches
                #                 if od[7] == 'B'
                #             ]

                #             a_origin_flights_C = [
                #                 (od, time)
                #                 for (od, time) in matches
                #                 if od[7] == 'C'
                #             ]

                #             for od, time in a_origin_flights_A:
                #                 flight_name = od[2]+od[7]+time
                #                 sigma_inflow_A.append(flight_name)

                #             for od, time in a_origin_flights_B:
                #                 flight_name = od[2]+od[7]+time
                #                 sigma_inflow_B.append(flight_name)

                #             for od, time in a_origin_flights_C:
                #                 flight_name = od[2]+od[7]+time
                #                 sigma_inflow_C.append(flight_name)

                #             for od, time in a_origin_flights_A_out:
                #                 flight_name = od[2]+od[7]+time
                #                 sigma_outflow_A.append(flight_name)

                #             for od, time in a_origin_flights_B_out:
                #                 flight_name = od[2]+od[7]+time
                #                 sigma_outflow_B.append(flight_name)

                #             for od, time in a_origin_flights_C_out:
                #                 flight_name = od[2]+od[7]+time
                #                 sigma_outflow_C.append(flight_name)

                #             od_list_f, time_list_f, product_list_f, pax_list_f, price_list_f, capacitycoeff_list_f = [], [], [], [], [], []
                #             od_list_l, time_list_l, product_list_l, pax_list_l, price_list_l, capacitycoeff_list_l = [], [], [], [], [], []
                #             pt = []
                #             value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                #             value_0_list, ratio_0_list = [], []
                #             for match in matches:
                #                 origin = match[0][2]
                #                 destination = match[0][7]
                #                 time = match[1]
                #                 od = str((origin, destination))
                #                 code = f"{origin}{destination}{time}"
                #                 pt.append(code)
                #                 retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                #                 doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_pax=")
                #                 for doc in doc_1:
                #                     content = doc.page_content
                #                     pattern = r',\s*(?=\w+=)' 
                #                     parts = re.split(pattern, content)

                #                     pairs = [p.strip().replace('"', "'") for p in parts]
                #                     for pair in pairs:
                #                         key, value = pair.split('=')
                #                         if key == 'OD':
                #                             od_list_f.append(value)
                #                         elif key == 'Departure Time':
                #                             time_list_f.append(value)
                #                         elif key == 'Oneway_Product':
                #                             product_list_f.append(value)
                #                         elif key == 'avg_pax':
                #                             pax_list_f.append(value)  
                #                         elif key == 'avg_price':
                #                             price_list_f.append(value) 

                #                 doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_pax=")
                #                 for doc in doc_2:
                #                     content = doc.page_content
                #                     pattern = r',\s*(?=\w+=)'  
                #                     parts = re.split(pattern, content)

                #                     pairs = [p.strip().replace('"', "'") for p in parts]
                #                     for pair in pairs:
                #                         key, value = pair.split('=')
                #                         if key == 'OD':
                #                             od_list_l.append(value)
                #                         elif key == 'Departure Time':
                #                             time_list_l.append(value)
                #                         elif key == 'Oneway_Product':
                #                             product_list_l.append(value)
                #                         elif key == 'avg_pax':
                #                             pax_list_l.append(value) 
                #                         elif key == 'avg_price':
                #                             price_list_l.append(value) 


                #                 value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(origin,time)
                #                 value_f_list.append(str(value_1))
                #                 ratio_f_list.append(str(ratio_1))
                #                 value_0_list.append(str(value_0))
                #                 ratio_0_list.append(str(ratio_0))
                #                 value_l_list.append(str(value_2))
                #                 ratio_l_list.append(str(ratio_2))

                #             doc =  f'pt = {pt} \n'
                #             doc +=  f'sigma_inflow_A = {sigma_inflow_A} \n sigma_outflow_A = {sigma_outflow_A}\n sigma_inflow_B = {sigma_inflow_B}\n sigma_outflow_B = {sigma_outflow_B}\n sigma_inflow_C = {sigma_inflow_C}\n sigma_outflow_C = {sigma_outflow_C} \n'
                            
                #             doc += f"\n avg_pax_f={pax_list_f} \n avg_pax_l={pax_list_l} \n avg_price_f={price_list_f}  \n avg_price_l={price_list_l} \n value_f_list ={value_f_list}\n  ratio_f_list={ratio_f_list}\n  value_l_list={value_l_list}\n  ratio_l_list={ratio_l_list}\n  value_0_list={value_0_list}\n  ratio_0_list={ratio_0_list}"
                #             doc += f"\n option_num = {num_flights} \n"
                #             doc += f"capacity_consum = {eco_flex_capacity}"
                            
                #             return doc


                #         def retrieve_key_information(query):
                #             matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                #             od = []
                #             t = []
                #             for match in matches:
                #                 origin = match[0][2]
                #                 destination = match[0][7]
                #                 time = match[1]
                #                 od_pair = str((origin, destination))
                #                 od.append(od_pair)
                #                 t.append(time)
                #             return od, t



                #         def retrieve_time_period(departure_time):
                #             intervals = {
                #                 '12pm~6pm': (time_class(12, 0), time_class(18, 0)),
                #                 '6pm~10pm': (time_class(18, 0), time_class(22, 0)),
                #                 '10pm~8am': (time_class(22, 0), time_class(8, 0)),
                #                 '8am~12pm': (time_class(8, 0), time_class(12, 0))
                #             }

                #             if isinstance(departure_time, str):
                #                 try:
                #                     hours, minutes = map(int, departure_time.split(':'))
                #                     departure_time = time_class(hours, minutes)
                #                 except ValueError:
                #                     raise ValueError("Time format should be: 'HH:MM'")

                #             for interval_name, (start, end) in intervals.items():
                #                 if start < end: 
                #                     if start <= departure_time < end:
                #                         return interval_name
                #                 else: 
                #                     if departure_time >= start or departure_time < end:
                #                         return interval_name

                #             return "Unknown"

                #         def retrieve_parameter(origin,time_interval,product):
                #             time_interval = f'({time_interval})'
                #             key = product + '*' + time_interval
                #             _value_ = 0
                #             _ratio_ = 0
                #             no_purchase_value = 0
                #             no_purchase_value_ratio = 0
                #             subset = v1[v1['Origin'] == origin]
                #             if key in subset.columns and not subset.empty:
                #                 _value_ = subset[key].values[0]
                #             subset2 = v2[v2['Origin'] == origin]
                #             if key in subset2.columns and not subset2.empty:
                #                 _ratio_ = subset2[key].values[0]

                #             if 'no_purchase' in subset.columns and not subset.empty:
                #                 no_purchase_value = subset['no_purchase'].values[0]
                #             if 'no_purchase' in subset2.columns and not subset2.empty:
                #                 no_purchase_value_ratio = subset2['no_purchase'].values[0]
                #             return _value_,_ratio_,no_purchase_value,no_purchase_value_ratio

                #         def generate_coefficients(origin,time):
                #             value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                #             value_0_list, ratio_0_list = [], []

                #             departure_time = datetime.strptime(time, '%H:%M').time()
                #             time_interval = retrieve_time_period(departure_time)
                #             value_1,ratio_1,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_flexi')

                #             value_2,ratio_2,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_lite')

                #             return value_1,ratio_1,value_2,ratio_2,value_0,ratio_0


                #         example_data_description = csv_qa_tool(problem_description)
                #         example_matches = retrieve_key_information(problem_description)
                #         fewshot_example = f'''
                #         Question: {problem_description}

                #         Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. The given OD and Departure Time values are {example_matches}

                #         Action: CSVQA

                #         Action Input: {problem_description}

                #         Observation: {example_data_description}

                #         Thought: Now I have known the answer.

                #         Final Answer:

                #         max \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                #         capacity_consum*x_f[i] + x_l[i] <= 187
                #         ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                #         x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #         x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                #         x_f[i] <= 10000* y[i]
                #         x_l[i] <= 10000* y[i]
                #         x_o[i] <= 10000* y[i]
                #         \sum_i \in sigma_inflow_A y[i] = \sum_i \in sigma_outflow_A y[i]
                #         \sum_i \in sigma_inflow_B y[i] = \sum_i \in sigma_outflow_B y[i]
                #         \sum_i \in sigma_inflow_C y[i] = \sum_i \in sigma_outflow_C y[i]
                #         \sum_i y[i] <= option_num
                #         x_f[i],x_l[i],x_o[i] >= 0
                #         y[i] is binary, where decision variables are based on the list pt and x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]. To be more specific, {example_data_description}
                #         '''

                #         tools = [Tool(name="CSVQA", func=csv_qa_tool, description="Retrieve flight data.")]

                #         llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=user_api_key)
                #         prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

                #                 Please refer to the following example and generate the answer in the same format:

                #                 {fewshot_example}

                #                 Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

                #                 When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

                #                 """

                #         suffix = """

                #                 Begin!

                #                 User Description: {input}
                #                 {agent_scratchpad}"""


                #         agent2 = initialize_agent(
                #             tools,
                #             llm=llm,
                #             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                #             agent_kwargs={
                #                 "prefix": prefix,
                #                 "suffix": suffix,
                #             },
                #             verbose=True,
                #             handle_parsing_errors=True
                #         )

                #         llm_code = ChatOpenAI(
                #                     temperature=0.0, model_name="gpt-4", openai_api_key=user_api_key
                #                 )

                #         def clean_text_preserve_newlines(text):
                #             cleaned = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
                #             cleaned = re.sub(r'[^\x20-\x7E\n]', '', cleaned)
                #             cleaned = re.sub(r'(\n\s+)(\w+\s*=)', r'\n\2', cleaned)
                #             cleaned = re.sub(r'\[\s+', '[', cleaned)
                #             cleaned = re.sub(r'\s+\]', ']', cleaned)
                #             cleaned = re.sub(r',\s+', ', ', cleaned)

                #             return cleaned

                #         def policy_sblp_small_model_code(query):
                #             old_stdout = sys.stdout
                #             sys.stdout = buffer = StringIO()
                #             result = agent2.invoke({"input": query})
                #             output_model = result['output']
                #             sys.stdout = old_stdout
                #             verbose_logs = buffer.getvalue()
                #             observations = re.findall(r"Observation: (.*?)(?=\nThought:|\nFinal Answer:)", verbose_logs, re.DOTALL)
                #             if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_model: 
                #                     format = '''
                #                     max \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                #                     capacity_consum*x_f[i] + x_l[i] <= 187
                #                     ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                #                     x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #                     x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                #                     x_f[i] <= 10000* y[i]
                #                     x_l[i] <= 10000* y[i]
                #                     x_o[i] <= 10000* y[i]
                #                     \sum_i \in sigma_inflow_A y[i] = \sum_i \in sigma_outflow_A y[i]
                #                     \sum_i \in sigma_inflow_B y[i] = \sum_i \in sigma_outflow_B y[i]
                #                     \sum_i \in sigma_inflow_C y[i] = \sum_i \in sigma_outflow_C y[i]
                #                     \sum_i y[i] <= option_num
                #                     x_f[i],x_l[i],x_o[i] >= 0
                #                     y[i] is binary, where decision variables are based on the list pt and x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]. To be more specific,
                #             '''
                #                     text = re.sub(r'\[\d+m', '', str(observations[0]))
                #                     output_model = format + text

                #             prompt = f"""
                #             You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

                #             Mathematical Optimization Model:
                #             {output_model}

                #             For example, here is a simple instance for reference:

                #             Mathematical Optimization Model:

                #             Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                #             Capacity Constraints:

                #             capacity_consum*x_f[i] + x_l[i] <= 187

                #             Balance Constraints:

                #             ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                #             Scale Constraints:
                #             x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #             x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                #             M Constraints:

                #             x_f[i] <= 10000* y[i]
                #             x_l[i] <= 10000* y[i]
                #             x_o[i] <= 10000* y[i]

                #             Flow Conservation Constraints:

                #             \sum_i \in sigma_inflow_A y[i] = \sum_i \in sigma_outflow_A y[i]

                #             \sum_i \in sigma_inflow_B y[i] = \sum_i \in sigma_outflow_B y[i]

                #             \sum_i \in sigma_inflow_C y[i] = \sum_i \in sigma_outflow_C y[i]

                #             Cardinality Constraints:

                #             \sum_i y[i] <= option_num

                #             Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                #             Binary Constraints: y[i] is binary, where decision variables are based on the list pt = ['CB10:45', 'AB11:20', 'BC11:20', 'BA12:25', 'BC12:25', 'CB12:35', 'AB12:40', 'AB12:55', 'AC13:05', 'AB13:55', 'CB14:15', 'CA14:15', 'BA14:25', 'BC14:25', 'AC15:45', 'AB16:05', 'BC16:30', 'CB16:55', 'CA16:55', 'AB17:05', 'AC17:25', 'AB18:00', 'CB18:30', 'CA18:30'], x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]
                #             sigma_inflow_A = ['BA12:25', 'CA14:15', 'BA14:25', 'CA16:55', 'CA18:30'], sigma_outflow_A = ['AB11:20', 'AB12:40', 'AB12:55', 'AC13:05', 'AB13:55', 'AC15:45', 'AB16:05', 'AB17:05', 'AC17:25', 'AB18:00'], sigma_inflow_B = ['CB10:45', 'AB11:20', 'CB12:35', 'AB12:40', 'AB12:55', 'AB13:55', 'CB14:15', 'AB16:05', 'CB16:55', 'AB17:05', 'AB18:00', 'CB18:30'], sigma_outflow_B = ['BC11:20', 'BA12:25', 'BC12:25', 'BA14:25', 'BC14:25', 'BC16:30'], sigma_inflow_C = ['BC11:20', 'BC12:25', 'AC13:05', 'BC14:25', 'AC15:45', 'BC16:30', 'AC17:25'], sigma_outflow_C = ['CB10:45', 'CB12:35', 'CB14:15', 'CA14:15', 'CB16:55', 'CA16:55', 'CB18:30', 'CA18:30']

                #             avg_pax_f=['127.86', '160.71', '133.14', '103.79', '36.0', '156.57', '87.36', '131.43', '111.64', '142.43', '33.71', '91.64', '133.0', '9.4', '133.71', '167.79', '92.36', '2.0', '149.29', '152.36', '130.14', '104.14', '13.57', '149.07']
                #             avg_pax_l=['15.5', '18.64', '14.5', '23.36', '3.79', '19.93', '25.0', '20.29', '23.79', '28.21', '5.93', '17.86', '28.93', '1.67', '20.5', '25.43', '6.36', '2.4', '24.5', '27.14', '19.14', '13.64', '3.38', '17.43']
                #             avg_price_f=['886.96', '1464.07', '873.59', '1479.64', '1520.65', '883.77', '1467.29', '856.66', '1663.16', '1477.78', '1510.9', '1638.05', '1480.24', '1521.05', '1669.65', '852.25', '907.06', '1583.0', '1665.33', '1475.07', '1670.39', '833.23', '1523.0', '1665.21']
                #             avg_price_l=['263.88', '441.88', '269.24', '470.28', '596.44', '299.39', '505.8', '279.25', '565.96', '464.67', '496.51', '531.5', '470.65', '821.73', '506.95', '275.1', '268.12', '558.48', '493.28', '461.18', '473.92', '261.39', '443.91', '463.64']
                #             value_f_list =['1.916', '2.05', '1.864', '3.079', '3.079', '2.826', '2.803', '2.803', '2.803', '2.803', '2.826', '2.826', '3.079', '3.079', '2.803', '2.803', '3.079', '2.826', '2.826', '2.803', '2.803', '1.624', '3.126', '3.126']
                #             ratio_f_list=['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72', '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72', '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']
                #             value_l_list=['1', '1', '1', '1.652', '1.652', '1.475', '1.367', '1.367', '1.367', '1.367', '1.475', '1.475', '1.652', '1.652', '1.367', '1.367', '1.652', '1.475', '1.475', '1.367', '1.367', '0.793', '1.631', '1.631']
                #             ratio_l_list=['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72', '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72', '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']
                #             value_0_list=['1.2', '0.9', '2.0', '2.0', '2.0', '1.2', '0.9', '0.9', '0.9', '0.9', '1.2', '1.2', '2.0', '2.0', '0.9', '0.9', '2.0', '1.2', '1.2', '0.9', '0.9', '0.9', '1.2', '1.2']
                #             ratio_0_list=['1.72', '4.92', '3.7', '3.7', '3.7', '1.72', '4.92', '4.92', '4.92', '4.92', '1.72', '1.72', '3.7', '3.7', '4.92', '4.92', '3.7', '1.72', '1.72', '4.92', '4.92', '4.92', '1.72', '1.72']
                #             option_num = 4

                #             capacity_consum = 3

                #             The corresponding Python code for this instance is as follows:

                #             ```python
                #             import gurobipy as gp
                #             from gurobipy import GRB

                #             # Create model
                #             model = gp.Model("FlightTicketOptimization")

                #             # Data
                #             pt = ['CB10:45', 'AB11:20', 'BC11:20', 'BA12:25', 'BC12:25', 'CB12:35', 'AB12:40',
                #                 'AB12:55', 'AC13:05', 'AB13:55', 'CB14:15', 'CA14:15', 'BA14:25', 'BC14:25',
                #                 'AC15:45', 'AB16:05', 'BC16:30', 'CB16:55', 'CA16:55', 'AB17:05', 'AC17:25',
                #                 'AB18:00', 'CB18:30', 'CA18:30']

                #             # Convert all string numbers to float
                #             avg_pax_f = [float(x) for x in ['127.86', '160.71', '133.14', '103.79', '36.0', '156.57', '87.36',
                #                                             '131.43', '111.64', '142.43', '33.71', '91.64', '133.0', '9.4',
                #                                             '133.71', '167.79', '92.36', '2.0', '149.29', '152.36', '130.14',
                #                                             '104.14', '13.57', '149.07']]

                #             avg_pax_l = [float(x) for x in ['15.5', '18.64', '14.5', '23.36', '3.79', '19.93', '25.0',
                #                                             '20.29', '23.79', '28.21', '5.93', '17.86', '28.93', '1.67',
                #                                             '20.5', '25.43', '6.36', '2.4', '24.5', '27.14', '19.14',
                #                                             '13.64', '3.38', '17.43']]

                #             avg_price_f = [float(x) for x in ['886.96', '1464.07', '873.59', '1479.64', '1520.65', '883.77',
                #                                             '1467.29', '856.66', '1663.16', '1477.78', '1510.9', '1638.05',
                #                                             '1480.24', '1521.05', '1669.65', '852.25', '907.06', '1583.0',
                #                                             '1665.33', '1475.07', '1670.39', '833.23', '1523.0', '1665.21']]

                #             avg_price_l = [float(x) for x in ['263.88', '441.88', '269.24', '470.28', '596.44', '299.39',
                #                                             '505.8', '279.25', '565.96', '464.67', '496.51', '531.5',
                #                                             '470.65', '821.73', '506.95', '275.1', '268.12', '558.48',
                #                                             '493.28', '461.18', '473.92', '261.39', '443.91', '463.64']]

                #             value_f_list = [float(x) for x in ['1.916', '2.05', '1.864', '3.079', '3.079', '2.826', '2.803',
                #                                             '2.803', '2.803', '2.803', '2.826', '2.826', '3.079', '3.079',
                #                                             '2.803', '2.803', '3.079', '2.826', '2.826', '2.803', '2.803',
                #                                             '1.624', '3.126', '3.126']]

                #             ratio_f_list = [float(x) for x in ['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72',
                #                                             '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72',
                #                                             '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']]

                #             value_l_list = [float(x) for x in ['1', '1', '1', '1.652', '1.652', '1.475', '1.367', '1.367',
                #                                             '1.367', '1.367', '1.475', '1.475', '1.652', '1.652', '1.367',
                #                                             '1.367', '1.652', '1.475', '1.475', '1.367', '1.367', '0.793',
                #                                             '1.631', '1.631']]

                #             ratio_l_list = [float(x) for x in ['0.97', '0.72', '0.9', '0.9', '0.9', '0.97', '0.72', '0.72',
                #                                             '0.72', '0.72', '0.97', '0.97', '0.9', '0.9', '0.72', '0.72',
                #                                             '0.9', '0.97', '0.97', '0.72', '0.72', '0.72', '0.97', '0.97']]

                #             value_0_list = [float(x) for x in ['1.2', '0.9', '2.0', '2.0', '2.0', '1.2', '0.9', '0.9', '0.9',
                #                                             '0.9', '1.2', '1.2', '2.0', '2.0', '0.9', '0.9', '2.0', '1.2',
                #                                             '1.2', '0.9', '0.9', '0.9', '1.2', '1.2']]

                #             ratio_0_list = [float(x) for x in ['1.72', '4.92', '3.7', '3.7', '3.7', '1.72', '4.92', '4.92',
                #                                             '4.92', '4.92', '1.72', '1.72', '3.7', '3.7', '4.92', '4.92',
                #                                             '3.7', '1.72', '1.72', '4.92', '4.92', '4.92', '1.72', '1.72']]

                #             # Flow conservation sets
                #             sigma_inflow_A = ['BA12:25', 'CA14:15', 'BA14:25', 'CA16:55', 'CA18:30']
                #             sigma_outflow_A = ['AB11:20', 'AB12:40', 'AB12:55', 'AC13:05', 'AB13:55', 'AC15:45',
                #                             'AB16:05', 'AB17:05', 'AC17:25', 'AB18:00']
                #             sigma_inflow_B = ['CB10:45', 'AB11:20', 'CB12:35', 'AB12:40', 'AB12:55', 'AB13:55',
                #                             'CB14:15', 'AB16:05', 'CB16:55', 'AB17:05', 'AB18:00', 'CB18:30']
                #             sigma_outflow_B = ['BC11:20', 'BA12:25', 'BC12:25', 'BA14:25', 'BC14:25', 'BC16:30']
                #             sigma_inflow_C = ['BC11:20', 'BC12:25', 'AC13:05', 'BC14:25', 'AC15:45', 'BC16:30', 'AC17:25']
                #             sigma_outflow_C = ['CB10:45', 'CB12:35', 'CB14:15', 'CA14:15', 'CB16:55', 'CA16:55',
                #                             'CB18:30', 'CA18:30']

                #             option_num = 4
                #             capacity_consum = 3

                #             # Decision variables
                #             x_f = model.addVars(pt, name="x_f")  # Flexible tickets
                #             x_l = model.addVars(pt, name="x_l")  # Limited tickets
                #             x_o = model.addVars(pt, name="x_o")  # Other tickets
                #             y = model.addVars(pt, vtype=GRB.BINARY, name="y")  # Flight selection

                #             # Objective: Maximize revenue
                #             model.setObjective(
                #                 gp.quicksum(avg_price_f[i] * x_f[pt[i]] + avg_price_l[i] * x_l[pt[i]] for i in range(len(pt))),
                #                 GRB.MAXIMIZE
                #             )

                #             # Capacity constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187, f"capacity_{{pt[i]}}")

                #             # Balance constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(
                #                     ratio_f_list[i] * x_f[pt[i]] + ratio_l_list[i] * x_l[pt[i]] + ratio_0_list[i] * x_o[pt[i]]
                #                     <= avg_pax_f[i] + avg_pax_l[i],
                #                     f"balance_{{pt[i]}}"
                #                 )

                #             # Scale constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(x_f[pt[i]]/value_f_list[i] - x_o[pt[i]]/value_0_list[i] <= 0, f"scale_f_{{pt[i]}}")
                #                 model.addConstr(x_l[pt[i]]/value_l_list[i] - x_o[pt[i]]/value_0_list[i] <= 0, f"scale_l_{{pt[i]}}")

                #             # M constraints (linking constraints)
                #             M = 10000
                #             for i in range(len(pt)):
                #                 model.addConstr(x_f[pt[i]] <= M * y[pt[i]], f"M_f_{{pt[i]}}")
                #                 model.addConstr(x_l[pt[i]] <= M * y[pt[i]], f"M_l_{{pt[i]}}")
                #                 model.addConstr(x_o[pt[i]] <= M * y[pt[i]], f"M_o_{{pt[i]}}")

                #             # Flow conservation constraints
                #             model.addConstr(
                #                 gp.quicksum(y[flight] for flight in sigma_inflow_A) ==
                #                 gp.quicksum(y[flight] for flight in sigma_outflow_A),
                #                 "flow_conservation_A"
                #             )

                #             model.addConstr(
                #                 gp.quicksum(y[flight] for flight in sigma_inflow_B) ==
                #                 gp.quicksum(y[flight] for flight in sigma_outflow_B),
                #                 "flow_conservation_B"
                #             )

                #             model.addConstr(
                #                 gp.quicksum(y[flight] for flight in sigma_inflow_C) ==
                #                 gp.quicksum(y[flight] for flight in sigma_outflow_C),
                #                 "flow_conservation_C"
                #             )

                #             # Cardinality constraint
                #             model.addConstr(gp.quicksum(y[pt[i]] for i in range(len(pt))) <= option_num, "cardinality")

                #             # Non-negativity constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(x_f[pt[i]] >= 0, f"nonneg_f_{{pt[i]}}")
                #                 model.addConstr(x_l[pt[i]] >= 0, f"nonneg_l_{{pt[i]}}")
                #                 model.addConstr(x_o[pt[i]] >= 0, f"nonneg_o_{{pt[i]}}")

                #             # Optimize model
                #             model.optimize()
                #             if model.status == GRB.OPTIMAL:
                #                 print("Optimal Solution Found:")
                #                 # Print decision variable values
                #                 for var in model.getVars():
                #                     if 'y' in var.VarName and var.X>0:
                #                         print(var.VarName,var.X)
                #             else:
                #                 print("No optimal solution found. Status Code:", m.status)

                #             """

                #             messages = [
                #                 HumanMessage(content=prompt)  
                #             ]

                #             response = llm_code(messages)
                #             output_code = response.content
                #             if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_code:  
                #                 format_code = '''
                #             import gurobipy as gp
                #             from gurobipy import GRB

                #             # Create model
                #             model = gp.Model("FlightTicketOptimization")
                #             x_f = model.addVars(pt, name="x_f")  # Flexible tickets
                #             x_l = model.addVars(pt, name="x_l")  # Limited tickets
                #             x_o = model.addVars(pt, name="x_o")  # Other tickets
                #             y = model.addVars(pt, vtype=GRB.BINARY, name="y")  # Flight selection

                #             # Objective: Maximize revenue
                #             model.setObjective(
                #                 gp.quicksum(avg_price_f[i] * x_f[pt[i]] + avg_price_l[i] * x_l[pt[i]] for i in range(len(pt))),
                #                 GRB.MAXIMIZE
                #             )

                #             # Capacity constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187, f"capacity_{{pt[i]}}")

                #             # Balance constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(
                #                     float(ratio_f_list[i]) * x_f[pt[i]] + float(ratio_l_list[i]) * x_l[pt[i]] + float(ratio_0_list[i]) * x_o[pt[i]]
                #                     <= float(avg_pax_f[i]) + float(avg_pax_l[i]),
                #                     f"balance_{{pt[i]}}"
                #                 )

                #             # Scale constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(x_f[pt[i]]/float(value_f_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_f_{{pt[i]}}")
                #                 model.addConstr(x_l[pt[i]]/float(value_l_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_l_{{pt[i]}}")

                #             # M constraints (linking constraints)
                #             M = 10000
                #             for i in range(len(pt)):
                #                 model.addConstr(x_f[pt[i]] <= M * y[pt[i]], f"M_f_{{pt[i]}}")
                #                 model.addConstr(x_l[pt[i]] <= M * y[pt[i]], f"M_l_{{pt[i]}}")
                #                 model.addConstr(x_o[pt[i]] <= M * y[pt[i]], f"M_o_{{pt[i]}}")

                #             # Flow conservation constraints
                #             model.addConstr(
                #                 gp.quicksum(y[flight] for flight in sigma_inflow_A) ==
                #                 gp.quicksum(y[flight] for flight in sigma_outflow_A),
                #                 "flow_conservation_A"
                #             )

                #             model.addConstr(
                #                 gp.quicksum(y[flight] for flight in sigma_inflow_B) ==
                #                 gp.quicksum(y[flight] for flight in sigma_outflow_B),
                #                 "flow_conservation_B"
                #             )

                #             model.addConstr(
                #                 gp.quicksum(y[flight] for flight in sigma_inflow_C) ==
                #                 gp.quicksum(y[flight] for flight in sigma_outflow_C),
                #                 "flow_conservation_C"
                #             )

                #             # Cardinality constraint
                #             model.addConstr(gp.quicksum(y[pt[i]] for i in range(len(pt))) <= option_num, "cardinality")

                #             # Non-negativity constraints
                #             for i in range(len(pt)):
                #                 model.addConstr(x_f[pt[i]] >= 0, f"nonneg_f_{{pt[i]}}")
                #                 model.addConstr(x_l[pt[i]] >= 0, f"nonneg_l_{{pt[i]}}")
                #                 model.addConstr(x_o[pt[i]] >= 0, f"nonneg_o_{{pt[i]}}")

                #             # Optimize model
                #             model.optimize()

                #             # Save model to file for inspection
                #             model.write("flight_optimization.lp")
                #             if model.status == GRB.OPTIMAL:
                #                 print("Optimal Solution Found:")
                #                 # Print decision variable values
                #                 for var in model.getVars():
                #                     if 'y' in var.VarName and var.X>0:
                #                         print(var.VarName,var.X)
                #             else:
                #                 print("No optimal solution found. Status Code:", m.status)
                #             '''
                #                 text = clean_text_preserve_newlines(str(observations[0]))
                #                 output_code = text + format_code
                #             return output_model,output_code
                        
                #         output_model, response = policy_sblp_small_model_code(query)
                #         ai_response = output_model
                        

                #     elif "recommend the optimal" in query:
                #         def csv_qa_tool(query: str):
                #             matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                #             num_match = re.search(r"optimal (\d+) flights", query)
                #             num_flights = int(num_match.group(1)) if num_match else None  # 3
                #             capacity_match = re.search(r"each Eco_flex ticket consumes (\d+) units of flight capacity", query)
                #             if capacity_match:
                #                 eco_flex_capacity = capacity_match.group(1)
                #             else:
                #                 eco_flex_capacity = 1.2
                #             od_list_f, time_list_f, product_list_f, pax_list_f, price_list_f, capacitycoeff_list_f = [], [], [], [], [], []
                #             od_list_l, time_list_l, product_list_l, pax_list_l, price_list_l, capacitycoeff_list_l = [], [], [], [], [], []
                #             pt = []
                #             value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                #             value_0_list, ratio_0_list = [], []
                #             for match in matches:
                #                 origin = match[0][2]
                #                 destination = match[0][7]
                #                 time = match[1]
                #                 od = str((origin, destination))
                #                 code = f"{origin}{destination}{time}"
                #                 pt.append(code)
                #                 retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                #                 doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_pax=")
                #                 for doc in doc_1:
                #                     content = doc.page_content
                #                     pattern = r',\s*(?=\w+=)'  
                #                     parts = re.split(pattern, content)

                #                     pairs = [p.strip().replace('"', "'") for p in parts]
                #                     for pair in pairs:
                #                         key, value = pair.split('=')
                #                         if key == 'OD':
                #                             od_list_f.append(value)
                #                         elif key == 'Departure Time':
                #                             time_list_f.append(value)
                #                         elif key == 'Oneway_Product':
                #                             product_list_f.append(value)
                #                         elif key == 'avg_pax':
                #                             pax_list_f.append(value)  
                #                         elif key == 'avg_price':
                #                             price_list_f.append(value) 
                #                         elif key == 'capacity_coefficient':
                #                             capacitycoeff_list_f.append(value) 

                #                 doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_pax=")
                #                 for doc in doc_2:
                #                     content = doc.page_content
                #                     pattern = r',\s*(?=\w+=)' 
                #                     parts = re.split(pattern, content)

                #                     pairs = [p.strip().replace('"', "'") for p in parts]
                #                     for pair in pairs:
                #                         key, value = pair.split('=')
                #                         if key == 'OD':
                #                             od_list_l.append(value)
                #                         elif key == 'Departure Time':
                #                             time_list_l.append(value)
                #                         elif key == 'Oneway_Product':
                #                             product_list_l.append(value)
                #                         elif key == 'avg_pax':
                #                             pax_list_l.append(value)  
                #                         elif key == 'avg_price':
                #                             price_list_l.append(value) 
                #                         elif key == 'capacity_coefficient':
                #                             capacitycoeff_list_l.append(value) 

                #                 value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(origin,time)
                #                 value_f_list.append(str(value_1))
                #                 ratio_f_list.append(str(ratio_1))
                #                 value_0_list.append(str(value_0))
                #                 ratio_0_list.append(str(ratio_0))
                #                 value_l_list.append(str(value_2))
                #                 ratio_l_list.append(str(ratio_2))

                #             doc =  f'decision variables are based on the list pt = {pt}, x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt] \n'
                #             doc += f"\n avg_pax_f={pax_list_f} \n avg_pax_l={pax_list_l} \n avg_price_f={price_list_f}  \n avg_price_l={price_list_l}  \n value_f_list ={value_f_list}\n  ratio_f_list={ratio_f_list}\n  value_l_list={value_l_list}\n  ratio_l_list={ratio_l_list}\n  value_0_list={value_0_list}\n  ratio_0_list={ratio_0_list}"
                #             doc += f"\n option_num = {num_flights} \n"
                #             doc += f"\n capacity_consum = {eco_flex_capacity} \n"
                #             return doc


                #         def retrieve_key_information(query):
                #             matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                #             od = []
                #             t = []
                #             for match in matches:
                #                 origin = match[0][2]
                #                 destination = match[0][7]
                #                 time = match[1]
                #                 od_pair = str((origin, destination))
                #                 od.append(od_pair)
                #                 t.append(time)
                #             return od, t

                #         def retrieve_time_period(departure_time):
                #             intervals = {
                #                 '12pm~6pm': (time_class(12, 0), time_class(18, 0)),
                #                 '6pm~10pm': (time_class(18, 0), time_class(22, 0)),
                #                 '10pm~8am': (time_class(22, 0), time_class(8, 0)),
                #                 '8am~12pm': (time_class(8, 0), time_class(12, 0))
                #             }

                #             if isinstance(departure_time, str):
                #                 try:
                #                     hours, minutes = map(int, departure_time.split(':'))
                #                     departure_time = time_class(hours, minutes)
                #                 except ValueError:
                #                     raise ValueError("'HH:MM'")

                #             for interval_name, (start, end) in intervals.items():
                #                 if start < end:  
                #                     if start <= departure_time < end:
                #                         return interval_name
                #                 else:  
                #                     if departure_time >= start or departure_time < end:
                #                         return interval_name

                #             return "Unknown"

                #         def retrieve_parameter(origin,time_interval,product):
                #             time_interval = f'({time_interval})'
                #             key = product + '*' + time_interval
                #             _value_ = 0
                #             _ratio_ = 0
                #             no_purchase_value = 0
                #             no_purchase_value_ratio = 0
                #             subset = v1[v1['Origin'] == origin]
                #             if key in subset.columns and not subset.empty:
                #                 _value_ = subset[key].values[0]
                #             subset2 = v2[v2['Origin'] == origin]
                #             if key in subset2.columns and not subset2.empty:
                #                 _ratio_ = subset2[key].values[0]

                #             if 'no_purchase' in subset.columns and not subset.empty:
                #                 no_purchase_value = subset['no_purchase'].values[0]
                #             if 'no_purchase' in subset2.columns and not subset2.empty:
                #                 no_purchase_value_ratio = subset2['no_purchase'].values[0]
                #             return _value_,_ratio_,no_purchase_value,no_purchase_value_ratio

                #         def generate_coefficients(origin,time):
                #             value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                #             value_0_list, ratio_0_list = [], []

                #             departure_time = datetime.strptime(time, '%H:%M').time()
                #             time_interval = retrieve_time_period(departure_time)
                #             value_1,ratio_1,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_flexi')

                #             value_2,ratio_2,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_lite')

                #             return value_1,ratio_1,value_2,ratio_2,value_0,ratio_0

                #         example_data_description = csv_qa_tool(problem_description)
                #         example_matches = retrieve_key_information(problem_description)
                #         fewshot_example = f'''
                #         Question: {problem_description}

                #         Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. The given OD and Departure Time values are {example_matches}

                #         Action: CSVQA

                #         Action Input: {problem_description}

                #         Observation: {example_data_description}

                #         Thought: Now I have known the answer.

                #         Final Answer:

                #         Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                #         Capacity Constraints:

                #         capacity_consum*x_f[i] + x_l[i] <= 187

                #         Balance Constraints:

                #         ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                #         Scale Constraints:
                #         x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #         x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                #         M Constraints:

                #         x_f[i] <= 10000* y[i]
                #         x_l[i] <= 10000* y[i]
                #         x_o[i] <= 10000* y[i]

                #         Cardinality Constraints:

                #         \sum_i y[i] <= option_num

                #         Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                #         Binary Constraints: y[i] is binary, where {example_data_description}
                #         '''

                #         tools = [Tool(name="CSVQA", func=csv_qa_tool, description="Retrieve flight data.")]

                #         llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=user_api_key)
                #         prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

                #                 Please refer to the following example and generate the answer in the same format:

                #                 {fewshot_example}

                #                 Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

                #                 When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

                #                 """

                #         suffix = """

                #                 Begin!

                #                 User Description: {input}
                #                 {agent_scratchpad}"""



                #         format = '''

                #         Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                #         Capacity Constraints:

                #         capacity_consum*x_f[i] + x_l[i] <= 187

                #         Balance Constraints:

                #         ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                #         Scale Constraints:
                #         x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #         x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0

                #         M Constraints:

                #         x_f[i] <= 10000* y[i]
                #         x_l[i] <= 10000* y[i]
                #         x_o[i] <= 10000* y[i]

                #         Cardinality Constraints:

                #         \sum_i y[i] <= option_num

                #         Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                #         Binary Constraints: y[i] \in {0,1}
                #         '''

                #         agent2 = initialize_agent(
                #             tools,
                #             llm=llm,
                #             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                #             agent_kwargs={
                #                 "prefix": prefix,
                #                 "suffix": suffix,
                #             },
                #             verbose=True,
                #             handle_parsing_errors=True
                #         )


                #         llm_code = ChatOpenAI(
                #         temperature=0.0, model_name="gpt-4", openai_api_key=user_api_key
                #         )

                #         def clean_text_preserve_newlines(text):
                #             cleaned = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
                #             cleaned = re.sub(r'[^\x20-\x7E\n]', '', cleaned)
                #             cleaned = re.sub(r'(\n\s+)(\w+\s*=)', r'\n\2', cleaned)
                #             cleaned = re.sub(r'\[\s+', '[', cleaned)
                #             cleaned = re.sub(r'\s+\]', ']', cleaned)
                #             cleaned = re.sub(r',\s+', ', ', cleaned)

                #             return cleaned

                #         def policy_sblp_small_model_code(query):
                #             old_stdout = sys.stdout
                #             sys.stdout = buffer = StringIO()
                #             result = agent2.invoke({"input": query})
                #             output_model = result['output']
                #             sys.stdout = old_stdout
                #             verbose_logs = buffer.getvalue()
                #             observations = re.findall(r"Observation: (.*?)(?=\nThought:|\nFinal Answer:)", verbose_logs, re.DOTALL)
                #             if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_model: 
                #                 format = '''
                #                 max \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                #                 capacity_consum*x_f[i] + x_l[i] <= 187
                #                 ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                #                 x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #                 x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                #                 x_f[i] <= 10000* y[i]
                #                 x_l[i] <= 10000* y[i]
                #                 x_o[i] <= 10000* y[i]
                #                 \sum_i y[i] <= option_num
                #                 x_f[i],x_l[i],x_o[i] >= 0
                #                 y[i] is binary, where decision variables are based on the list pt and x_f = [x_code_f for code in pt], x_l = [x_code_l for code in pt], x_o = [x_code_o for code in pt], y = [y_code for code in pt]. To be more specific,
                #             '''
                #                 text = re.sub(r'\[\d+m', '', str(observations[0]))
                #                 output_model = format + text


                #             prompt = f"""
                #         You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

                #         Mathematical Optimization Model:
                #         {output_model}

                #         For example, here is a simple instance for reference:

                #         Mathematical Optimization Model:

                #         Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                #         Capacity Constraints:

                #         capacity_consum*x_f[i] + x_l[i] <= 187

                #         Balance Constraints:

                #         ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                #         Scale Constraints:
                #         x_f[i]/ratio_f_list[i] - x_o[i]/value_0_list[i] <=0
                #         x_l[i]/ratio_l_list[i] - x_o[i]/value_0_list[i] <=0

                #         M Constraints:

                #         x_f[i] <= 10000* y[i]
                #         x_l[i] <= 10000* y[i]
                #         x_o[i] <= 10000* y[i]

                #         Cardinality Constraints:

                #         \sum_i y[i] <= option_num

                #         Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                #         Binary Constraints: y[i] \in {{0,1}}

                #         , where decision variables are:
                #         pt = ['BC12:25', 'BC14:25', 'BC15:40', 'BC09:05', 'BC11:20', 'BC16:30', 'BC19:05']
                #         avg_pax_f=['36.0', '9.4', '11.69', '17.5', '133.14', '92.36', '137.14']
                #         avg_pax_l=['3.79', '1.67', '3.27', '3.23', '14.5', '6.36', '17.57']
                #         avg_price_f=['1520.65', '1521.05', '1544.13', '1489.26', '873.59', '907.06', '875.17']
                #         avg_price_l=['596.44', '821.73', '483.3', '443.62', '269.24', '268.12', '282.71']
                #         value_f_list =[np.float64(3.079), np.float64(3.079), np.float64(3.079), np.float64(1.864), np.float64(1.864), np.float64(3.079), np.float64(2.411)]
                #         ratio_f_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                #         value_l_list=[np.float64(1.652), np.float64(1.652), np.float64(1.652), np.int64(1), np.int64(1), np.float64(1.652), np.float64(1.293)]
                #         ratio_l_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                #         value_0_list=[np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0)]
                #         ratio_0_list=[np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7)]
                #         capacity_consum = 1.2


                #         The corresponding Python code for this instance is as follows:

                #         ```python
                #         import gurobipy as gp
                #         from gurobipy import GRB
                #         import numpy as np

                #         pt = ['BC12:25', 'BC14:25', 'BC15:40', 'BC09:05', 'BC11:20', 'BC16:30', 'BC19:05']
                #         avg_pax_f=['36.0', '9.4', '11.69', '17.5', '133.14', '92.36', '137.14']
                #         avg_pax_l=['3.79', '1.67', '3.27', '3.23', '14.5', '6.36', '17.57']
                #         avg_price_f=['1520.65', '1521.05', '1544.13', '1489.26', '873.59', '907.06', '875.17']
                #         avg_price_l=['596.44', '821.73', '483.3', '443.62', '269.24', '268.12', '282.71']
                #         value_f_list =[np.float64(3.079), np.float64(3.079), np.float64(3.079), np.float64(1.864), np.float64(1.864), np.float64(3.079), np.float64(2.411)]
                #         ratio_f_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                #         value_l_list=[np.float64(1.652), np.float64(1.652), np.float64(1.652), np.int64(1), np.int64(1), np.float64(1.652), np.float64(1.293)]
                #         ratio_l_list=[np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9), np.float64(0.9)]
                #         value_0_list=[np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0), np.float64(2.0)]
                #         ratio_0_list=[np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7), np.float64(3.7)]
                #         capacity_consum = 1.2

                #         pax_f = []
                #         for i in avg_pax_f:
                #             i = float(i)
                #             pax_f.append(i)

                #         pax_l = []
                #         for i in avg_pax_l:
                #             pax_l.append(float(i))

                #         price_f = []
                #         for i in avg_price_f:
                #             price_f.append(float(i))

                #         price_l = []
                #         for i in avg_price_l:
                #             price_l.append(float(i))

                #         coefficient_f = []

                #         for i in capacity_coefficient_f:
                #             coefficient_f.append(float(i))

                #         coefficient_l = []

                #         for i in capacity_coefficient_l:
                #             coefficient_l.append(float(i))
                #         # Create the model
                #         m = gp.Model("Flight_Optimization")

                #         # Decision variables
                #         x_f = m.addVars(pt, vtype=GRB.CONTINUOUS, name="x_f")
                #         x_l = m.addVars(pt, vtype=GRB.CONTINUOUS, name="x_l")
                #         x_o = m.addVars(pt, vtype=GRB.CONTINUOUS, name="x_o")
                #         y = m.addVars(pt, vtype=GRB.BINARY, name="y")

                #         # Objective function
                #         m.setObjective(
                #             gp.quicksum(float(avg_price_f[i]) * x_f[pt[i]] + float(avg_price_l[i]) * x_l[pt[i]] for i in range(7)),
                #             GRB.MAXIMIZE
                #         )

                #         # Constraints
                #         for i in range(7):
                #             m.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187)
                #             m.addConstr(x_f[pt[i]] <= 10000 * y[pt[i]])
                #             m.addConstr(x_l[pt[i]] <= 10000 * y[pt[i]])
                #             m.addConstr(x_o[pt[i]] <= 10000 * y[pt[i]])

                #             m.addConstr(ratio_f_list[i] * x_f[pt[i]] + ratio_l_list[i] * x_l[pt[i]] + ratio_0_list[i] * x_o[pt[i]] <= float(avg_pax_f[i]) + float(avg_pax_l[i]))

                #             m.addConstr(x_f[pt[i]] / value_f_list[i] <= x_o[pt[i]] / value_0_list[i])
                #             m.addConstr(x_l[pt[i]] / value_l_list[i] <= x_o[pt[i]] / value_0_list[i])


                #         # Solve the model
                #         m.optimize()
                #         m.write('test.lp')

                #         # Check if the model was solved successfully
                #         if m.status == GRB.OPTIMAL:
                #             print("Optimal Solution Found:")
                #             # Print decision variable values
                #             for var in m.getVars():
                #                 print(var.VarName,var.X)
                #         else:
                #             print("No optimal solution found. Status Code:", m.status)

                #         """

                #             messages = [
                #                 HumanMessage(content=prompt)  
                #             ]



                #             response = llm_code(messages)
                #             output_code = response.content

                #             if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" or "option_num" or "capacity_consum" not in output_code:  
                #                 format_code = '''
                #         import gurobipy as gp
                #         from gurobipy import GRB

                #         # Create model
                #         model = gp.Model("FlightTicketOptimization")
                #         x_f = model.addVars(pt, name="x_f")  # Flexible tickets
                #         x_l = model.addVars(pt, name="x_l")  # Limited tickets
                #         x_o = model.addVars(pt, name="x_o")  # Other tickets
                #         y = model.addVars(pt, vtype=GRB.BINARY, name="y")  # Flight selection

                #         # Objective: Maximize revenue
                #         model.setObjective(
                #             gp.quicksum(avg_price_f[i] * x_f[pt[i]] + avg_price_l[i] * x_l[pt[i]] for i in range(len(pt))),
                #             GRB.MAXIMIZE
                #         )

                #         # Capacity constraints
                #         for i in range(len(pt)):
                #             model.addConstr(capacity_consum * x_f[pt[i]] + x_l[pt[i]] <= 187, f"capacity_{{pt[i]}}")

                #         # Balance constraints
                #         for i in range(len(pt)):
                #             model.addConstr(
                #                 float(ratio_f_list[i]) * x_f[pt[i]] + float(ratio_l_list[i]) * x_l[pt[i]] + float(ratio_0_list[i]) * x_o[pt[i]]
                #                 <= float(avg_pax_f[i]) + float(avg_pax_l[i]),
                #                 f"balance_{{pt[i]}}"
                #             )

                #         # Scale constraints
                #         for i in range(len(pt)):
                #             model.addConstr(x_f[pt[i]]/float(value_f_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_f_{{pt[i]}}")
                #             model.addConstr(x_l[pt[i]]/float(value_l_list[i]) - x_o[pt[i]]/float(value_0_list[i]) <= 0, f"scale_l_{{pt[i]}}")

                #         # M constraints (linking constraints)
                #         M = 10000
                #         for i in range(len(pt)):
                #             model.addConstr(x_f[pt[i]] <= M * y[pt[i]], f"M_f_{{pt[i]}}")
                #             model.addConstr(x_l[pt[i]] <= M * y[pt[i]], f"M_l_{{pt[i]}}")
                #             model.addConstr(x_o[pt[i]] <= M * y[pt[i]], f"M_o_{{pt[i]}}")

                #         # Cardinality constraint
                #         model.addConstr(gp.quicksum(y[pt[i]] for i in range(len(pt))) <= option_num, "cardinality")

                #         # Non-negativity constraints
                #         for i in range(len(pt)):
                #             model.addConstr(x_f[pt[i]] >= 0, f"nonneg_f_{{pt[i]}}")
                #             model.addConstr(x_l[pt[i]] >= 0, f"nonneg_l_{{pt[i]}}")
                #             model.addConstr(x_o[pt[i]] >= 0, f"nonneg_o_{{pt[i]}}")

                #         # Optimize model
                #         model.optimize()

                #         # Save model to file for inspection
                #         model.write("flight_optimization.lp")
                #         if model.status == GRB.OPTIMAL:
                #             print("Optimal Solution Found:")
                #             # Print decision variable values
                #             for var in model.getVars():
                #                 if 'y' in var.VarName and var.X>0:
                #                     print(var.VarName,var.X)
                #         else:
                #             print("No optimal solution found. Status Code:", m.status)
                #         '''
                #             text = clean_text_preserve_newlines(str(observations[0]))
                #             output_code = text + format_code

                #             return output_model,output_code

                #         output_model, response = policy_sblp_small_model_code(query)
                #         ai_response = output_model



                #     else:
                #         def csv_qa_tool(query: str):
                #             matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                #             num_match = re.search(r"optimal (\d+) flights", query)
                #             num_flights = int(num_match.group(1)) if num_match else None  
                #             od_list_f, time_list_f, product_list_f, pax_list_f, price_list_f = [], [], [], [], []
                #             od_list_l, time_list_l, product_list_l, pax_list_l, price_list_l = [], [], [], [], []
                #             pt = []
                #             value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                #             value_0_list, ratio_0_list = [], []
                #             for match in matches:
                #                 origin = match[0][2]
                #                 destination = match[0][7]
                #                 time = match[1]
                #                 od = str((origin, destination))
                #                 code = f"{origin}{destination}{time}"
                #                 pt.append(code)
                #                 retriever = new_vectors.as_retriever(search_kwargs={'k': 1,"filter": {"OD": od, "time": time}})
                #                 doc_1= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_flexi, avg_pax=")
                #                 for doc in doc_1:
                #                     content = doc.page_content
                #                     pattern = r',\s*(?=\w+=)' 
                #                     parts = re.split(pattern, content)

                #                     pairs = [p.strip().replace('"', "'") for p in parts]
                #                     for pair in pairs:
                #                         key, value = pair.split('=')
                #                         if key == 'OD':
                #                             od_list_f.append(value)
                #                         elif key == 'Departure Time':
                #                             time_list_f.append(value)
                #                         elif key == 'Oneway_Product':
                #                             product_list_f.append(value)
                #                         elif key == 'avg_pax':
                #                             pax_list_f.append(value) 
                #                         elif key == 'avg_price':
                #                             price_list_f.append(value) 


                #                 doc_2= retriever.get_relevant_documents(f"OD={od}, Departure Time={time}, Oneway_Product=Eco_lite, avg_pax=")
                #                 for doc in doc_2:
                #                     content = doc.page_content
                #                     pattern = r',\s*(?=\w+=)' 
                #                     parts = re.split(pattern, content)

                #                     pairs = [p.strip().replace('"', "'") for p in parts]
                #                     for pair in pairs:
                #                         key, value = pair.split('=')
                #                         if key == 'OD':
                #                             od_list_l.append(value)
                #                         elif key == 'Departure Time':
                #                             time_list_l.append(value)
                #                         elif key == 'Oneway_Product':
                #                             product_list_l.append(value)
                #                         elif key == 'avg_pax':
                #                             pax_list_l.append(value)
                #                         elif key == 'avg_price':
                #                             price_list_l.append(value) 

                #                 value_1,ratio_1,value_2,ratio_2,value_0,ratio_0 = generate_coefficients(origin,time)
                #                 value_f_list.append(str(value_1))
                #                 ratio_f_list.append(str(ratio_1))
                #                 value_0_list.append(str(value_0))
                #                 ratio_0_list.append(str(ratio_0))
                #                 value_l_list.append(str(value_2))
                #                 ratio_l_list.append(str(ratio_2))

                #             doc = f'Decision Variables are based on list I, I= {pt}'
                #             doc += f"\n avg_pax_f={pax_list_f} \n avg_pax_l={pax_list_l} \n avg_price_f={price_list_f}  \n avg_price_l={price_list_l} \n value_f_list ={value_f_list}\n  ratio_f_list={ratio_f_list}\n  value_l_list={value_l_list}\n  ratio_l_list={ratio_l_list}\n  value_0_list={value_0_list}\n  ratio_0_list={ratio_0_list}"
                #             return doc


                #         def retrieve_key_information(query):
                #             matches = re.findall(r"\(OD\s*=\s*(\(\s*'[^']+'\s*,\s*'[^']+'\s*\))\s+AND\s+Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)", query)
                #             od = []
                #             t = []
                #             for match in matches:
                #                 origin = match[0][2]
                #                 destination = match[0][7]
                #                 time = match[1]
                #                 od_pair = str((origin, destination))
                #                 od.append(od_pair)
                #                 t.append(time)
                #             return od, t


                #         def retrieve_time_period(departure_time):
                #             intervals = {
                #                 '12pm~6pm': (time_class(12, 0), time_class(18, 0)),
                #                 '6pm~10pm': (time_class(18, 0), time_class(22, 0)),
                #                 '10pm~8am': (time_class(22, 0), time_class(8, 0)),
                #                 '8am~12pm': (time_class(8, 0), time_class(12, 0))
                #             }

                #             if isinstance(departure_time, str):
                #                 try:
                #                     hours, minutes = map(int, departure_time.split(':'))
                #                     departure_time = time_class(hours, minutes)
                #                 except ValueError:
                #                     raise ValueError("'HH:MM'")

                #             for interval_name, (start, end) in intervals.items():
                #                 if start < end: 
                #                     if start <= departure_time < end:
                #                         return interval_name
                #                 else:  
                #                     if departure_time >= start or departure_time < end:
                #                         return interval_name

                #             return "Unknown"

                #         def retrieve_parameter(origin,time_interval,product):
                #             time_interval = f'({time_interval})'
                #             key = product + '*' + time_interval
                #             _value_ = 0
                #             _ratio_ = 0
                #             no_purchase_value = 0
                #             no_purchase_value_ratio = 0
                #             subset = v1[v1['Origin'] == origin]
                #             if key in subset.columns and not subset.empty:
                #                 _value_ = subset[key].values[0]
                #             subset2 = v2[v2['Origin'] == origin]
                #             if key in subset2.columns and not subset2.empty:
                #                 _ratio_ = subset2[key].values[0]

                #             if 'no_purchase' in subset.columns and not subset.empty:
                #                 no_purchase_value = subset['no_purchase'].values[0]
                #             if 'no_purchase' in subset2.columns and not subset2.empty:
                #                 no_purchase_value_ratio = subset2['no_purchase'].values[0]
                #             return _value_,_ratio_,no_purchase_value,no_purchase_value_ratio

                #         def generate_coefficients(origin,time):
                #             value_f_list, ratio_f_list, value_l_list, ratio_l_list = [], [], [], []
                #             value_0_list, ratio_0_list = [], []

                #             departure_time = datetime.strptime(time, '%H:%M').time()
                #             time_interval = retrieve_time_period(departure_time)
                #             value_1,ratio_1,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_flexi')

                #             value_2,ratio_2,value_0,ratio_0 = retrieve_parameter(origin,time_interval,'Eco_lite')

                #             return value_1,ratio_1,value_2,ratio_2,value_0,ratio_0


                #         example_data_description = csv_qa_tool(problem_description)
                #         example_matches = retrieve_key_information(problem_description)
                #         fewshot_example = f'''
                #         Question: {problem_description}

                #         Thought: I need to retrieve relevant information from 'information1.csv' for the given OD and Departure Time values. Next I need to retrieve the relevant coefficients from v1 and v2 based on the retrieved ticket information. The given OD and Departure Time values are {example_matches}

                #         Action: CSVQA

                #         Action Input: {problem_description}

                #         Observation: {example_data_description}

                #         Thought: Now I have known the answer.

                #         Final Answer:

                #         Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )

                #         Capacity Constraints:

                #         1.2*x_f[i] + x_l[i] <= 187

                #         Balance Constraints:

                #         ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]

                #         Scale Constraints:
                #         x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #         x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0


                #         Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0

                #         , where {example_data_description}
                #         '''

                #         tools = [Tool(name="CSVQA", func=csv_qa_tool, description="Retrieve flight data.")]
                #         llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=user_api_key)
                #         prefix = f"""You are an assistant that generates a mathematical model based on the user's description and provided CSV data.

                #                 Please refer to the following example and generate the answer in the same format:

                #                 {fewshot_example}

                #                 Note: Please retrieve all neccessary information from the CSV file to generate the answer. When you generate the answer, please output required parameters in a whole text, including all vectors and matrices.

                #                 When you need to retrieve information from the CSV file, and write SBLP formulation by using the provided tools.

                #                 """

                #         suffix = """

                #                 Begin!

                #                 User Description: {input}
                #                 {agent_scratchpad}"""


                #         agent2 = initialize_agent(
                #             tools,
                #             llm=llm,
                #             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                #             agent_kwargs={
                #                 "prefix": prefix,
                #                 "suffix": suffix,
                #             },
                #             verbose=True,
                #             handle_parsing_errors=True
                #         )

                #         def conversational_chat(query):
                #             result = agent2.invoke({"input": query})
                #             output = result['output']
                #             return output

                #         def clean_text_preserve_newlines(text):
                #             cleaned = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
                #             cleaned = re.sub(r'[^\x20-\x7E\n]', '', cleaned)
                #             cleaned = re.sub(r'(\n\s+)(\w+\s*=)', r'\n\2', cleaned)
                #             cleaned = re.sub(r'\[\s+', '[', cleaned)
                #             cleaned = re.sub(r'\s+\]', ']', cleaned)
                #             cleaned = re.sub(r',\s+', ', ', cleaned)

                #             return cleaned


                #         def get_answer(query):
                #             old_stdout = sys.stdout
                #             sys.stdout = buffer = StringIO()
                #             result = agent2.invoke({"input": query})
                #             output_model = result['output']
                #             sys.stdout = old_stdout
                #             verbose_logs = buffer.getvalue()
                #             observations = re.findall(r"Observation: (.*?)(?=\nThought:|\nFinal Answer:)", verbose_logs, re.DOTALL)
                #             if "avg_pax_f" or "avg_pax_l" or  "avg_price_f" or "capacity_list_f" or "value_list_f" or "value_0_list" or "ratio_0_list" not in output_model: 
                #                 format = '''
                #                 Objective Function: Maximize \sum_i (avg_price_f[i]*x_f[i] + avg_price_l[i]*x_l[i] )
                #                 Capacity Constraints:
                #                 1.2*x_f[i] + x_l[i] <= 187
                #                 Balance Constraints:
                #                 ratio_f_list[i]* x_f[i] +  ratio_l_list[i]* x_l[i] + ratio_0_list[i]* x_o[i] <= avg_pax_f[i]  + avg_pax_l [i]
                #                 Scale Constraints:
                #                 x_f[i]/value_f_list[i] - x_o[i]/value_0_list[i] <=0
                #                 x_l[i]/value_l_list[i] - x_o[i]/value_0_list[i] <=0
                #                 Nonnegative Constraints: x_f[i],x_l[i],x_o[i] >= 0,where
                #                 '''
                #                 text = re.sub(r'\[\d+m', '', str(observations[0]))
                #                 output_model = format + clean_text_preserve_newlines(text)
                            
                #             return output_model

                #         result = get_answer(query)
                #         ai_response = result


            # 记录并显示AI响应
    #        st.session_state.generated.append(ai_response)
        # 显示AI响应
        with container:
#            message(ai_response, key=f"ai_{len(st.session_state.generated)}")
            message(ai_response)

        # except Exception as e:
        #     error_msg = f"⚠️ Error processing request: {str(e)}"
        #     st.session_state.generated.append(error_msg)
        #     with container:
        #         message(error_msg, key=f"error_{len(st.session_state.generated)}")


    # 聊天界面布局 ======================================================
    def render_chat_history(container):
        # 仅渲染已有历史记录（应对页面刷新）
        with container:
            for i in range(len(st.session_state.past)):
                # 用户消息
                message(
                    st.session_state.past[i],
                    is_user=True,
                    key=f"hist_user_{i}"
                )
                # AI响应（检查索引边界）
                if i < len(st.session_state.generated):
                    message(
                        st.session_state.generated[i],
                        key=f"hist_ai_{i}"
                    )



    # 用户输入表单 ======================================================
    with input_container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="What optimization problem do you want to ask? Please type in the Chatbox in detail",
                height=150,
                key='input'
            )
            submit_button = st.form_submit_button(label='Send')

    # 处理用户提交 ======================================================
    if submit_button and user_input:
        process_user_input(user_input, agent_pc, response_container, dfs)

    # 历史消息渲染
#    render_chat_history(response_container)








