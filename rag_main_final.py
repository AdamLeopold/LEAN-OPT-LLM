# pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
from streamlit_chat import message
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
import re
from datetime import datetime, time

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

uploaded_files = st.sidebar.file_uploader("upload", type="csv", accept_multiple_files=True)

if uploaded_files:
    dfs = []  # 用于存储读取的 DataFrames
    for uploaded_file in uploaded_files:
        # 临时保存每个文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 读取 CSV 文件并添加到列表中
        df = pd.read_csv(tmp_file_path)
        dfs.append((uploaded_file.name, df))

    # 在侧边栏添加选择框
    problem_options = [
        "Please select an option",  # Placeholder option
        "Network Revenue Management Problem",
        "SBLP Singapore Airline Use Case",
        "Resource Allocation Problem",
        "Knapsack Problem",
        "Transportation Problem",
        "Minimum-Cost Flow Problem",
        "Assignment Problem",
        "Capacitated Facility Location Problem",
        "Uncapacitated Facility Location Problem",
    ]

    selected_problem = st.sidebar.selectbox("Please select your question: ", problem_options,index=0)

    # Check if the user has selected a valid option
    if selected_problem == "Please select an option":
        st.warning("Please select a problem from the dropdown.")
    else:
        # 根据用户的选择，执行相应的代码
        if selected_problem == "Network Revenue Management Problem":

            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
    #            st.session_state['generated'] = []
                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = []
    #            st.session_state['past'] = ["Hey ! 👋"]

            # container for the chat history
            response_container = st.container()
            # container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_area("Query:", placeholder=f"What {selected_problem} do you want to ask :)", key='input', height=150)
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    # **修改2：在调用生成回答之前，先添加用户输入并显示**
                    st.session_state['past'].append(user_input)

                    # 在界面上立即显示用户的问题
                    with response_container:
                        message(user_input, is_user=True, key=str(len(st.session_state['past']) - 1) + '_user',
                                avatar_style="big-smile")

                    # 显示一个进度条或加载动画
                    with st.spinner('Generating response...'):

                        information = pd.read_csv('NRM_example/nike Shoes Sales.csv')
                        information_head = information[:36]

                        # 将示例数据转换为字符串，供 few_shot_examples 使用
                        example_data_description = "\nHere is the product data:\n"
                        for index, row in information_head.iterrows():
                            example_data_description += f"Product {index + 1}: {row['Product Name']}, revenue w_{index + 1} = {row['Revenue']}, demand rate a_{index + 1} = {row['Demand']}, initial inventory c_{index + 1} = {row['Initial Inventory']}\n"

                        # 构建 problem_description 和 label
                        problem_description0 = problem_description = """The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer."""

                        problem_description += example_data_description

                        problem_description1 = 'The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories of the Nike x Olivia Kim brand, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” Moreover, the trade will be started only when the total demand is no less than 100 to ensure the trading efficiency. The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer.'

                        label = """
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
                         startup_constraint:
                         x_0+x_1+x_2+x_3 >=100
                        """

                        print(example_data_description)

                        few_shot_examples = f"""

                        Question: {problem_description1}

                        Based on the above description and data, please formulate a linear programming model.

                        Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file.

                        Action: CSVQA

                        Action Input: "Retrieve the product data related to Nike x OliviaKim to formulate the linear programming model."

                        Observation: {example_data_description}

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

                            # 遍历 DataFrame 的每一行并生成描述
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

                        memory = ConversationBufferMemory(memory_key="chat_history")


                        def conversational_chat(query):

                            result = agent.invoke(query)
                            st.session_state['history'].append((query, result['output']))

                            return result['output']


                        output = conversational_chat(user_input)

                    st.session_state['generated'].append(output)

                    # 显示 AI 的回答
                    with response_container:
                        message(output, key=str(len(st.session_state['generated']) - 1), avatar_style="thumbs")

                    # **修改3：在用户提交后，立即显示聊天记录**
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            # 已经在上面显示过最新的消息，这里避免重复显示
                            if i < len(st.session_state['past']) - 1:
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                                        avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")











        elif selected_problem == "Resource Allocation Problem":

            informationC = pd.read_csv('RA_example/Capacity.csv')
            informationP = pd.read_csv('RA_example/Products.csv')
            information=[]
            information.append(informationC)
            information.append(informationP)


            # # 将示例数据转换为字符串，供 few_shot_examples 使用
            example_data_description = "\nHere is the data:\n"
            # for index, row in informationC.iterrows():
            #     example_data_description += f"platform id {index + 1}: {row['platform_id']}, capacity = {row['capacity']}\n"
            #
            # for index, row in informationP.iterrows():
            #     example_data_description += f"genre {index + 1}: {row['genre']}, value = {row['value']}, memory requirement = {row['memory requirement']}\n"


            # 遍历每个 DataFrame 及其索引
            for df_index, df in enumerate(information):
                if df_index==0:
                    example_data_description += f"\nDataFrame {df_index + 1} - Capacity\n"
                elif df_index==1:
                    example_data_description += f"\nDataFrame {df_index + 1} - Products\n"

                # 遍历 DataFrame 的每一行并生成描述
                for index, row in df.iterrows():
                    description=""
#                    description = f"Row {index + 1}: "
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    example_data_description += description + "\n"



            # 构建 problem_description 和 label
            problem_description0 = problem_description = """A digital game store needs to decide which games to list on different platforms, considering that these games belong to various brands such as Sony, Amazon, and Apple. Each platform has a limited memory capacity, with specific details provided in "Capacity.csv." The predefined value and memory requirement of each game are available in "Products.csv.", with the column name being "value" and "memory requirement". The objective is to determine which genres and how many units of each game to list on each platform to maximize the total value of the games across all platforms, while ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  x_ij represent the number of units of games from genres j to be listed on platform i. For example, x_12 denotes the number of units of a specific game product (e.g., Sony Alpha Refrigerator) to be listed on Platform 1."""
            problem_description += example_data_description

            problem_description1 = 'A supermarket needs to allocate various products, including high-demand items like the Sony Alpha Refrigerator, Sony Bravia XR, and Sony PlayStation 5, across different retail shelves. The product values and space requirements are provided in the "Products.csv" dataset. Additionally, the store has multiple shelves, each with a total space limit and specific space constraints for Sony and Apple products, as outlined in the "Capacity.csv" file. The goal is to determine the optimal number of units of each Sony product to place on each shelf to maximize total value while ensuring that the space used by Sony products on each shelf does not exceed the brand-specific limits. The decision variables x_ij represent the number of units of product i to be placed on shelf j.'
            # label_head1 = """
            #         Maximize
            #            11197 x_0 + 9097 x_1 + 11197 x_2 + 9995 x_3
            #         Subject To
            #          inventory_constraint: x_0 <= 97
            #          demand_constraint: x_0 <= 17
            #          x_1 <= 240
            #          x_1 <= 26
            #          x_2 <= 322
            #          x_2 <= 50
            #          x_3 <= 281
            #          x_3 <= 53
            #         """

            with open('RA_example/Sony.txt', 'r', encoding='utf-8') as file:
                label = file.read()

            print(example_data_description)

            few_shot_examples = f"""
    
                    Question: {problem_description1}
    
                    Based on the above description and data, please formulate a linear programming model.
    
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

            # 遍历每个 DataFrame 及其索引
            for df_index, (file_name, df) in enumerate(dfs):
                # 将文件名添加到描述中
                data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

                # 遍历 DataFrame 的每一行并生成描述
                for index, row in df.iterrows():
                    description = ""
#                    description = f"Row {index + 1}: "
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    data.append(description + "\n")

            # for line in data:
            #     st.text(line)  # 使用 Streamlit 显示内容
            #st.text(few_shot_examples)



            # df = pd.read_csv(tmp_file_path)
            #
            # # 创建嵌入和向量存储
            # data = []
            # for index, row in df.iterrows():
            #     content = f"Product Name: {row['Product Name']}, Revenue: {row['Revenue']}, Demand: {row['Demand']}, Initial Inventory: {row['Initial Inventory']}"
            #     data.append(content)

            documents = [content for content in data]
            embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
            vectors = FAISS.from_texts(documents, embeddings)

            num_documents = len(documents)

            # 创建检索器和 RetrievalQA 链
            retriever = vectors.as_retriever(search_kwargs={'k': 20})
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
            # user_description = 'A digital game store needs to decide which games to list on different platforms, considering that these games belong to various genres such as racing, sports, and others. Each platform has a limited memory capacity, with specific details provided in "capacity.csv." The predefined value and memory requirement of each game are available in "products.csv.", with the column name being "value" and "memory requirement". The objective is to determine which genres and how many units of each game to list on each platform to maximize the total value of the games across all platforms, while ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  x_ij represent the number of units of games from genres j (e.g.,sports) to be listed on platform i. For example, x_12 denotes the number of units of a specific genre of games (e.g., sports) to be listed on Platform 1.'

            # 运行 Agent 并获取答案
            # answer = agent.invoke(user_description)
            # print("Answer:", answer['output'])

            memory = ConversationBufferMemory(memory_key="chat_history")


            #    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, memory=memory, verbose=True)

            def query_data(query):
                response = agent.invoke(query)
                return response


            def conversational_chat(query):

                result = agent.invoke(query)
                st.session_state['history'].append((query, result['output']))

                return result['output']


            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                #            st.session_state['generated'] = []
                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = []
            #            st.session_state['past'] = ["Hey ! 👋"]

            # container for the chat history
            response_container = st.container()
            # container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_area("Query:", placeholder=f"What {selected_problem} do you want to ask :)",
                                              key='input', height=150)
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    # **修改2：在调用生成回答之前，先添加用户输入并显示**
                    st.session_state['past'].append(user_input)

                    # 在界面上立即显示用户的问题
                    with response_container:
                        message(user_input, is_user=True, key=str(len(st.session_state['past']) - 1) + '_user',
                                avatar_style="big-smile")

                    # 显示一个进度条或加载动画
                    with st.spinner('Generating response...'):
                        output = conversational_chat(user_input)

                    st.session_state['generated'].append(output)

                    # 显示 AI 的回答
                    with response_container:
                        message(output, key=str(len(st.session_state['generated']) - 1), avatar_style="thumbs")

                    # **修改3：在用户提交后，立即显示聊天记录**
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            # 已经在上面显示过最新的消息，这里避免重复显示
                            if i < len(st.session_state['past']) - 1:
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                                        avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")












        elif selected_problem == "Knapsack Problem":

            informationC = pd.read_csv('KP_example/KPcapacity.csv')
            informationP = pd.read_csv('KP_example/KPproducts.csv')
            information = []
            information.append(informationC)
            information.append(informationP)

            # # 将示例数据转换为字符串，供 few_shot_examples 使用
            example_data_description = "\nHere is the data:\n"
            # for index, row in informationC.iterrows():
            #     example_data_description += f"platform id {index + 1}: {row['platform_id']}, capacity = {row['capacity']}\n"
            #
            # for index, row in informationP.iterrows():
            #     example_data_description += f"genre {index + 1}: {row['genre']}, value = {row['value']}, memory requirement = {row['memory requirement']}\n"

            # 遍历每个 DataFrame 及其索引
            for df_index, df in enumerate(information):
                if df_index == 0:
                    example_data_description += f"\nDataFrame {df_index + 1} - Capacity\n"
                elif df_index == 1:
                    example_data_description += f"\nDataFrame {df_index + 1} - Products\n"

                # 遍历 DataFrame 的每一行并生成描述
                for index, row in df.iterrows():
                    description = ""
                    #                    description = f"Row {index + 1}: "
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    example_data_description += description + "\n"

            # 构建 problem_description 和 label
            problem_description0 = problem_description = """Amazon needs to allocate different types of air conditioning units across multiple warehouse storage areas. Each storage area has a specific capacity limit, which is provided in the "KPcapacity.csv". The value and size of each type of air conditioner are listed in the "KPproduct.csv". The objective is to determine the optimal number of each air conditioner type to store in each warehouse to maximize the total value across all storage areas. At the same time, the total space occupied by the units in each warehouse must not exceed its capacity limit. Decision variables represent the quantity of each air conditioner type stored in each warehouse."""
            problem_description += example_data_description

            problem_description1 = 'Amazon needs to allocate different types of air conditioning units across multiple warehouse storage areas. Each storage area has a specific capacity limit, which is provided in the "KPcapacity.csv". The value and size of each type of air conditioner are listed in the "KPproduct.csv". The objective is to determine the optimal number of each air conditioner type to store in each warehouse to maximize the total value across all storage areas. At the same time, the total space occupied by the units in each warehouse must not exceed its capacity limit. Decision variables represent the quantity of each air conditioner type stored in each warehouse.'

            with open('KP_example/AmazonProductsSalesDataset2023.txt', 'r', encoding='utf-8') as file:
                label = file.read()

            few_shot_examples = f"""

                    Question: {problem_description1}

                    Based on the above description and data, please formulate a linear programming model.

                    Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Particularly, I should fighure out which product and how many of them in total in the CSV should be considered.

                    Action: CSVQA

                    Action Input: "Retrieve the capacity data and products data to formulate the linear programming model."

                    Observation: {example_data_description}

                    Thought: Now that I have the necessary data, I can construct the objective function and constraints. And I should generate the answer only using the format below. The expressions should not be simplified or abbreviated. I need to retrieve products similar to 'Nike x Olivia Kim' from the CSV file to formulate the linear programming model.

                    Final Answer: 
                    {label}
                    """

            # 加载实际的 CSV 文件
            data = []

            # 遍历每个 DataFrame 及其索引
            for df_index, (file_name, df) in enumerate(dfs):
                # 将文件名添加到描述中
                data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

                # 遍历 DataFrame 的每一行并生成描述
                for index, row in df.iterrows():
                    description = ""
                    #                    description = f"Row {index + 1}: "
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    data.append(description + "\n")

            # for line in data:
            #     st.text(line)  # 使用 Streamlit 显示内容
            # st.text(few_shot_examples)

            # df = pd.read_csv(tmp_file_path)
            #
            # # 创建嵌入和向量存储
            # data = []
            # for index, row in df.iterrows():
            #     content = f"Product Name: {row['Product Name']}, Revenue: {row['Revenue']}, Demand: {row['Demand']}, Initial Inventory: {row['Initial Inventory']}"
            #     data.append(content)

            documents = [content for content in data]
            embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
            vectors = FAISS.from_texts(documents, embeddings)

            num_documents = len(documents)

            # 创建检索器和 RetrievalQA 链
            retriever = vectors.as_retriever(search_kwargs={'k': 40})
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
            # user_description = 'A digital game store needs to decide which games to list on different platforms, considering that these games belong to various genres such as racing, sports, and others. Each platform has a limited memory capacity, with specific details provided in "capacity.csv." The predefined value and memory requirement of each game are available in "products.csv.", with the column name being "value" and "memory requirement". The objective is to determine which genres and how many units of each game to list on each platform to maximize the total value of the games across all platforms, while ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  x_ij represent the number of units of games from genres j (e.g.,sports) to be listed on platform i. For example, x_12 denotes the number of units of a specific genre of games (e.g., sports) to be listed on Platform 1.'

            # 运行 Agent 并获取答案
            # answer = agent.invoke(user_description)
            # print("Answer:", answer['output'])

            memory = ConversationBufferMemory(memory_key="chat_history")


            #    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, memory=memory, verbose=True)

            def query_data(query):
                response = agent.invoke(query)
                return response


            def conversational_chat(query):

                result = agent.invoke(query)
                st.session_state['history'].append((query, result['output']))

                return result['output']


            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                #            st.session_state['generated'] = []
                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = []
            #            st.session_state['past'] = ["Hey ! 👋"]

            # container for the chat history
            response_container = st.container()
            # container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_area("Query:",
                                                placeholder=f"What {selected_problem} do you want to ask :)",
                                                key='input', height=150)
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    # **修改2：在调用生成回答之前，先添加用户输入并显示**
                    st.session_state['past'].append(user_input)

                    # 在界面上立即显示用户的问题
                    with response_container:
                        message(user_input, is_user=True, key=str(len(st.session_state['past']) - 1) + '_user',
                                avatar_style="big-smile")

                    # 显示一个进度条或加载动画
                    with st.spinner('Generating response...'):
                        output = conversational_chat(user_input)

                    st.session_state['generated'].append(output)

                    # 显示 AI 的回答
                    with response_container:
                        message(output, key=str(len(st.session_state['generated']) - 1), avatar_style="thumbs")

                    # **修改3：在用户提交后，立即显示聊天记录**
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            # 已经在上面显示过最新的消息，这里避免重复显示
                            if i < len(st.session_state['past']) - 1:
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                                        avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")











        elif selected_problem == "Transportation Problem":

            informationCustomer = pd.read_csv('TP_example/customer_demand.csv')
            informationSupply = pd.read_csv('TP_example/supply_capacity.csv')
            informationCost = pd.read_csv('TP_example/transportation_costs.csv')
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
                for index, row in df.iterrows():
                    description = ""
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    example_data_description += description + "\n"

            # 构建 problem_description 和 label
            problem_description0 = problem_description = """Amazon has two distribution centers that need to supply essential goods daily to several customer groups. Each distribution center has a daily supply limit, with capacity data provided in “supply_capacity.csv.” Each customer group has a daily demand for these goods, with demand data detailed in “customer_demands.csv.” The transportation cost per unit of goods from each Amazon distribution center to each customer group is recorded in “transportation_costs.csv.” The objective is to determine the quantity of goods to be transported from each distribution center to each customer group, ensuring that demand is met while minimizing the total transportation cost. The decision variables  x_ij  represent the number of units of goods transported from Amazon distribution center  S_i  to customer group  C_j . For instance,  x_12  denotes the number of units of goods transported from Amazon distribution center  S_1  to customer group  C_2 . """
            problem_description += example_data_description

            problem_description1 = 'Amazon has two distribution centers that need to supply essential goods daily to several customer groups. Each distribution center has a daily supply limit, with capacity data provided in “supply_capacity.csv.” Each customer group has a daily demand for these goods, with demand data detailed in “customer_demands.csv.” The transportation cost per unit of goods from each Amazon distribution center to each customer group is recorded in “transportation_costs.csv.” The objective is to determine the quantity of goods to be transported from each distribution center to each customer group, ensuring that demand is met while minimizing the total transportation cost. The decision variables  x_ij  represent the number of units of goods transported from Amazon distribution center  S_i  to customer group  C_j . For instance,  x_12  denotes the number of units of goods transported from Amazon distribution center  S_1  to customer group  C_2 . '


            with open('TP_example/transportation_problem2.txt', 'r', encoding='utf-8') as file:
                label = file.read()

            few_shot_examples = f"""

                    Question: {problem_description1}

                    Based on the above description and data, please formulate a linear programming model.

                    Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Particularly, I should fighure out which product and how many of them in total in the CSV should be considered.

                    Action: CSVQA

                    Action Input: "Retrieve the capacity data and products data to formulate the linear programming model."

                    Observation: {example_data_description}

                    Thought: Now that I have the necessary data, I can construct the objective function and constraints. And I should generate the answer only using the format below. The expressions should not be simplified or abbreviated. I need to retrieve products similar to 'Nike x Olivia Kim' from the CSV file to formulate the linear programming model.

                    Final Answer: 
                    {label}
                    """

            # 加载实际的 CSV 文件
            data = []

            # 遍历每个 DataFrame 及其索引
            for df_index, (file_name, df) in enumerate(dfs):
                # 将文件名添加到描述中
                data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

                # 遍历 DataFrame 的每一行并生成描述
                for index, row in df.iterrows():
                    description = ""
                    #                    description = f"Row {index + 1}: "
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    data.append(description + "\n")

            for line in data:
                st.text(line)  # 使用 Streamlit 显示内容
            # st.text(few_shot_examples)

            # df = pd.read_csv(tmp_file_path)
            #
            # # 创建嵌入和向量存储
            # data = []
            # for index, row in df.iterrows():
            #     content = f"Product Name: {row['Product Name']}, Revenue: {row['Revenue']}, Demand: {row['Demand']}, Initial Inventory: {row['Initial Inventory']}"
            #     data.append(content)

            documents = [content for content in data]
            embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
            vectors = FAISS.from_texts(documents, embeddings)

            num_documents = len(documents)

            # 创建检索器和 RetrievalQA 链
            retriever = vectors.as_retriever(search_kwargs={'k': 80})
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
            # user_description = 'A digital game store needs to decide which games to list on different platforms, considering that these games belong to various genres such as racing, sports, and others. Each platform has a limited memory capacity, with specific details provided in "capacity.csv." The predefined value and memory requirement of each game are available in "products.csv.", with the column name being "value" and "memory requirement". The objective is to determine which genres and how many units of each game to list on each platform to maximize the total value of the games across all platforms, while ensuring that the total memory usage on each platform does not exceed its capacity. The decision variables  x_ij represent the number of units of games from genres j (e.g.,sports) to be listed on platform i. For example, x_12 denotes the number of units of a specific genre of games (e.g., sports) to be listed on Platform 1.'

            # 运行 Agent 并获取答案
            # answer = agent.invoke(user_description)
            # print("Answer:", answer['output'])

            memory = ConversationBufferMemory(memory_key="chat_history")


            #    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, memory=memory, verbose=True)

            def query_data(query):
                response = agent.invoke(query)
                return response


            def conversational_chat(query):

                result = agent.invoke(query)
                st.session_state['history'].append((query, result['output']))

                return result['output']


            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                #            st.session_state['generated'] = []
                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = []
            #            st.session_state['past'] = ["Hey ! 👋"]

            # container for the chat history
            response_container = st.container()
            # container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_area("Query:",
                                                placeholder=f"What {selected_problem} do you want to ask :)",
                                                key='input', height=150)
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    # **修改2：在调用生成回答之前，先添加用户输入并显示**
                    st.session_state['past'].append(user_input)

                    # 在界面上立即显示用户的问题
                    with response_container:
                        message(user_input, is_user=True, key=str(len(st.session_state['past']) - 1) + '_user',
                                avatar_style="big-smile")

                    # 显示一个进度条或加载动画
                    with st.spinner('Generating response...'):
                        output = conversational_chat(user_input)

                    st.session_state['generated'].append(output)

                    # 显示 AI 的回答
                    with response_container:
                        message(output, key=str(len(st.session_state['generated']) - 1), avatar_style="thumbs")

                    # **修改3：在用户提交后，立即显示聊天记录**
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            # 已经在上面显示过最新的消息，这里避免重复显示
                            if i < len(st.session_state['past']) - 1:
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                                        avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")








        elif selected_problem == "SBLP Singapore Airline Use Case":

            v1 = pd.read_csv('SBLP_example/AttractiveValues.csv', index_col=None)
            v2 = pd.read_csv('SBLP_example/adjusted_AttractiveValues.csv', index_col=None)
            v1 = v1.set_index(v1.columns[0])
            v2 = v2.set_index(v2.columns[0])
            v_br = v2 / v1
            v1 = v1.reset_index()
            v2 = v2.reset_index()

            information = pd.read_csv('SBLP_example/ticket_choices.csv')
            # 将示例数据转换为字符串，供 few_shot_examples 使用
            example_data_description = "\nHere is the flight data:\n"
            for index, row in information.iterrows():
                example_data_description += f"{index + 1}: POS = {row['POS']}, departure_time t_{index + 1} = {row['Departure_Time_Flight1']}, Oneway_Product a_{index + 1} = {row['Oneway_Product']}, avg_pax d_{index + 1} = {row['avg_pax']}, avf_price p_{index + 1} = {row['avg_price']}, capacity c_{index + 1} = {row['capacity']} \n"


            def ticket_analysis(ticket_info: str):
                """当你需要根据检索出的ticket信息，判断这些ticket的POS、departure_time区间、Oneway_Product，并从v1 和 v2中查找对应值时，请使用此工具。"""
                global v1, v2
                # Initialize lists to store results
                product_values = []
                no_purchase_values = []
                no_purchase_value_ratios = []

                # Split the ticket_info string into lines and filter out empty lines
                tickets = [line.strip() for line in ticket_info.strip().split('\n') if line.strip()]

                for ticket in tickets:
                    # 提取 POS、departure_time、Oneway_Product
                    pos_match = re.search(r'POS\s*=\s*([A-Z])', ticket)
                    time_match = re.search(r'departure_time\s*[t\d]*\s*=\s*([\d:]+)', ticket)
                    product_match = re.search(r'Oneway_product\s*[a\d]*\s*=\s*(\w+)', ticket)

                    if not pos_match or not time_match or not product_match:
                        pos_match = re.search(r'POS\s*=\s*([A-Z])', ticket, re.IGNORECASE)
                        time_match = re.search(r'Departure\s*Time\s*=\s*([\d:]+)', ticket, re.IGNORECASE)
                        product_match = re.search(r'Product\s*=\s*(\w+)', ticket, re.IGNORECASE)
                        pos_match = re.search(r'POS\s*=\s*([A-Z])', ticket)
                        time_match = re.search(r'Departure\s*Time\s*=\s*([\d:]+)', ticket)
                        product_match = re.search(r'Product\s*=\s*(\w+)', ticket)

                    pos = pos_match.group(1)
                    departure_time_str = time_match.group(1)
                    product = product_match.group(1)
                    # Convert departure_time_str to a Python time object
                    departure_time = datetime.strptime(departure_time_str, '%H:%M').time()

                    # Define time intervals
                    intervals = {
                        '12pm~6pm': (time(12, 0), time(18, 0)),
                        '6pm~10pm': (time(18, 0), time(22, 0)),
                        '10pm~8am': (time(22, 0), time(8, 0)),
                        '8am~12pm': (time(8, 0), time(12, 0))
                    }

                    # Determine the time interval for the departure time
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

                    # If no time interval is matched, skip this ticket
                    if time_interval is None:
                        continue

                    # Construct the column key based on Product and Time Interval
                    key = product + '*(' + time_interval + ')'

                    # Retrieve the value from v1
                    subset = v1[v1['POS'] == pos]
                    if key in subset.columns and not subset.empty:
                        value = subset[key].values[0]
                        product_values.append(value)
                    else:
                        print(f"Warning: Key '{key}' not found in v1 for POS '{pos}'.")

                    # Retrieve the value from v2
                    subset2 = v2[v2['POS'] == pos]
                    if key in subset2.columns and not subset2.empty:
                        value2 = subset2[key].values[0]
                        product_values.append(value2)
                    else:
                        print(f"Warning: Key '{key}' not found in v2 for POS '{pos}'.")

                    # Add no_purchase values for v1 (once per POS)
                    if 'no_purchase' in subset.columns and not subset.empty:
                        no_purchase_value = subset['no_purchase'].values[0]
                        if no_purchase_value not in no_purchase_values:
                            no_purchase_values.append(no_purchase_value)

                    # Add no_purchase value ratios for v2 (once per POS)
                    if 'no_purchase' in subset2.columns and not subset2.empty:
                        no_purchase_value_ratio = subset2['no_purchase'].values[0]
                        if no_purchase_value_ratio not in no_purchase_value_ratios:
                            no_purchase_value_ratios.append(no_purchase_value_ratio)

                # Combine all values into a single list
                result_values = product_values + no_purchase_values + no_purchase_value_ratios

                # Return the combined list of values
                return result_values


            coe_tool = Tool(
                name="coeff_retriever",
                func=ticket_analysis,
                description="retriever the relevant coefficient from dataframe v1 and dataframe v2"
            )

            # 加载实际的 CSV 文件
            data = []

            # 遍历每个 DataFrame 及其索引
            for df_index, (file_name, df) in enumerate(dfs):

                # data.append(f'\nDataFrame {df_index + 1} - ticket_choices\n')

                # 遍历 DataFrame 的每一行并生成描述
                for index, row in df.iterrows():
                    description = ""
                    description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
                    data.append(description + "\n")


            few_shot_examples = [
                f"""
                Question: For all flight ticket choices detailed in 'ticket_choice.csv',and the attraction values ratios are in the dataframe v2, and the attraction values are in the dataframe v1. write the sales-based linear formulation including objective functions, capacity constraints, balance constraints, scale constraints and nonnegative constraints for all flight ticket choices with POS A and departure time at 7:40.

                Thought: I need to retrieve relevant information of all the qualified ticket choices.

                Action: CSVQA

                Action Input: Retrieve the flight ticket choices information.

                Observation:
                {example_data_description}
                Of which the useful ticket choices are:
                POS = A, departure_time t2 = 7:40, Oneway_product a2 = Eco_flexi, avg_pax d2 = 5.351, avg_price p2 = 1630.291, capacity c2 = 79 ;
                POS = A, departure_time t3 = 7:40, Oneway_product a3 = Eco_lite, avg_pax d3 = 1.488, avg_price p3 = 483.509, capacity c_3 = 88 .

                Thought: Based on the previous observation, I now need to retrieve the attraction values from v1 and the attraction value ratios from v2 for the flight tickets and no_purchase choice based on the time range, POS, and Oneway_Product information using tools "coeff_retriever".

                Action: coeff_retriever

                Action Input:
                POS = A, departure_time t2 = 7:40, Oneway_product a2 = Eco_flexi, avg_pax d2 = 5.351, avg_price p2 = 1630.291, capacity c2 = 79 ;
                POS = A, departure_time t3 = 7:40, Oneway_product a3 = Eco_lite, avg_pax d3 = 1.488, avg_price p3 = 483.509, capacity c_3 = 88 .

                Observation:
                2.671, 0.740, 1.303, 0.361, 0.9, 0.249
                The attraction value and attraction value ratio retrieved for POS A, time 7:40, and Product Eco_flexi are 2.671 and 0.740,
                The attraction value and attraction value ratio retrieved for POS A, time 7:40, and Product Eco_lite are 1.303 and 0.361,
                The attraction value and attraction value ratio of no_purchase at POS A are 0.9 and 0.249;

                Final Answer:
                OBJECTIVE FUNCTION:
                    Max 1630.291 x_Af + 483.509 x_Al



                CONSTRAINTS:
                
                Capacity constraint: 
                    x_Af <= 79
                    x_Al <= 88
                    
                Balance constraint:
                    0.740 x_Af + 0.361 x_Al + 0.249 x_Ao = 5.351+1.488
                    
                Scale constraint:
                    x_Af/2.61 <= x_Ao/0.9
                    x_Al/1.303 <= x_Ao/0.9
                    
                Nonnegativity: 
                    x_Af>=0
                    x_Al>=0
            """,
            f"""
            Question: For all flight ticket choices detailed in 'ticket_choice.csv',and the attraction values ratios are in the dataframe v2, and the attraction values are in the dataframe v1. write the sales-based linear formulation including objective functions, capacity constraints, balance constraints, scale constraints and nonnegative constraints for all flight ticket choices with POS C and departure time at 09:05 and all flight ticket choices with POS B and departure time at 11:20.

            
            Thought: I need to retrieve relevant information of all the qualified ticket choices.

            Action: CSVQA

            Action Input: Retrieve the flight ticket choices information.

            Observation:
            {example_data_description}
            Of which the useful ticket choices are:
            POS = C, Departure Time = 09:05, Product = Eco_flexi, Average Passengers = 1.423, Average Price = 1494.511, Capacity = 13
            POS = B, Departure Time = 11:20, Product = Eco_flexi, Average Passengers = 2.339, Average Price = 1439.146, Capacity = 118
            POS = C, Departure Time = 09:05, Product = Eco_lite, Average Passengers = 1.25, Average Price = 409.85, Capacity = 138
            POS = B, Departure Time = 11:20, Product = Eco_lite, Average Passengers = 1.077, Average Price = 483.846, Capacity = 77


            Thought:Based on the previous observation, I now need to retrieve the attraction values from v1 and the attraction value ratios from v2 for the flight tickets and no_purchase choice based on the time range, POS, and Oneway_Product information using tools "coeff_retriever".

            Action: coeff_retriever

            Action Input:
            POS = C, Departure Time = 09:05, Product = Eco_flexi, Average Passengers = 1.423, Average Price = 1494.511, Capacity = 13
            POS = B, Departure Time = 11:20, Product = Eco_flexi, Average Passengers = 2.339, Average Price = 1439.146, Capacity = 118
            POS = C, Departure Time = 09:05, Product = Eco_lite, Average Passengers = 1.25, Average Price = 409.85, Capacity = 138
            POS = B, Departure Time = 11:20, Product = Eco_lite, Average Passengers = 1.077, Average Price = 483.846, Capacity = 77 
            
            Observation: [1.916, 0.061, 1.864, 0.19, 1, 0.032, 1, 0.102, 1.2, 2.0, 0.038, 0.204]
            Thought:I now know the final answer
            
            Final Answer:
            
            OBJECTIVE FUNCTION:
            Max  483.846 x_Bl + 1439.146 x_Bf + 409.85 x_Cl + 1494.511 x_Cf
            
            CONSTRAINTS:
            
            Capacity constraint: 
            x_Cf <= 13
            x_Bf <= 118
            x_Cl <= 138
            x_Bl <= 77
            
            Balance constraint:
            0.061 x_Cf + 0.032 x_Cl + 0.038 x_Co  = 1.423  + 1.25 
            0.19 x_Bf + 0.102 x_Bl + 0.204 x_Bo = 2.339 + 1.077
            
            Scale constraint:
            x_Cf/1.916 <= x_Co/1.2
            x_Bf/1.864 <= x_Bo/2.0
            x_Cl/1 <= x_Co/1.2
            x_Bl/1 <= x_Bo/2.0
            
            Nonnegativity: 
            x_Cf>=0
            x_Bf>=0
            x_Cl>=0
            x_Bl>=0

            """

            ]

            documents = [content for content in data]
            embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
            vectors = FAISS.from_texts(documents, embeddings)

            num_documents = len(documents)

            # 创建检索器和 RetrievalQA 链
            retriever = vectors.as_retriever(search_kwargs={'k': 20})
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


                    {few_shot_examples}

                    When you need to retrieve information from the CSV file, use the provided tool.

                    """

            suffix = """

                    Begin!

                    User Description: {input}
                    {agent_scratchpad}"""

            # 初始化 Agent
            agent = initialize_agent(
                tools=[qa_tool, coe_tool],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={
                    "prefix": prefix,
                    "suffix": suffix,
                },
                verbose=True,
                handle_parsing_errors=True,
            )

            def conversational_chat(query):

                result = agent.invoke(query)
                st.session_state['history'].append((query, result['output']))

                return result['output']


            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                #            st.session_state['generated'] = []
                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = []
            #            st.session_state['past'] = ["Hey ! 👋"]

            # container for the chat history
            response_container = st.container()
            # container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_area("Query:",
                                                placeholder=f"What {selected_problem} do you want to ask :)",
                                                key='input', height=150)
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    # **修改2：在调用生成回答之前，先添加用户输入并显示**
                    st.session_state['past'].append(user_input)

                    # 在界面上立即显示用户的问题
                    with response_container:
                        message(user_input, is_user=True, key=str(len(st.session_state['past']) - 1) + '_user',
                                avatar_style="big-smile")

                    # 显示一个进度条或加载动画
                    with st.spinner('Generating response...'):
                        output = conversational_chat(user_input)

                    st.session_state['generated'].append(output)

                    # 显示 AI 的回答
                    with response_container:
                        message(output, key=str(len(st.session_state['generated']) - 1), avatar_style="thumbs")

                    # **修改3：在用户提交后，立即显示聊天记录**
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            # 已经在上面显示过最新的消息，这里避免重复显示
                            if i < len(st.session_state['past']) - 1:
                                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                                        avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")






        else:
            st.write(f"You choose {selected_problem}. This type of problem is under construction :(")
# streamlit run gptrag_2.py