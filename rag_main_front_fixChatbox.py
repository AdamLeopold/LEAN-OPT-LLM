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


    # Initialize the LLM
    llm1 = ChatOpenAI(
        temperature=0.0, model_name="gpt-4o", openai_api_key=user_api_key
    )

    # Load and process the data
    loader = CSVLoader(file_path="RAG_doc2.csv", encoding="utf-8")
    data = loader.load()

    # Each line is a document
    documents = data

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    vectors = FAISS.from_documents(documents, embeddings)

    # Create a retriever
    retriever = vectors.as_retriever()

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm1,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


    # Create a tool using the RetrievalQA chain
    qa_tool = Tool(
        name="FileQA",
        func=qa_chain.invoke,
        description=(
            "Use this tool to answer questions about the problem type of the text. "
            "Provide the question as input, and the tool will retrieve the relevant information from the file and use it to answer the question."
        ),
    )

    # Define few-shot examples as a string
    few_shot_examples = """
    
    Question: What is the problem type in operation of the text? Please give the answer directly. Text:There are three best-selling items (P1, P2, P3) on Amazon with the profit w_1,w_2,w_3.There is an independent demand stream for each of the products. The objective of the company is to decide which demands to be fufilled over a ﬁnite sales horizon [0,10] to maximize the total expected revenue from ﬁxed initial inventories. The on-hand inventories for the three items are c_1,c_2,c_3 respectively. During the sales horizon, replenishment is not allowed and there is no any in-transit inventories. Customers who want to purchase P1,P2,P3 arrive at each period accoring to a Poisson process with a_1,a_2,a_3 the arrival rates respectively. Decision variables y_1,y_2,y_3 correspond to the number of requests that the firm plans to fulfill for product 1,2,3. These variables are all positive integers.
    
    Thought: I need to determine the problem type of the text. I'll use the FileQA tool to retrieve the relevant information.
    
    Action: FileQA
    
    Action Input: "What is the problem type in operation of the text? text:There are three best-selling items (P1, P2, P3) on Amazon with the profit w_1, w_2, w_3. ..."
    
    Observation: The problem type of the text is Network Revenue Management.
    
    Final Answer: Network Revenue Management.
    
    """

    # Create the prefix and suffix for the agent's prompt
    prefix = f"""You are a helpful assistant that can answer questions about operation problems. 
    
    Use the following examples as a guide. Always use the FileQA tool when you need to retrieve information from the file:
    
    
    {few_shot_examples}
    
    When you need to find information from the file, use the provided tools. And answer the question by given the answer directly. For example,
    
    """

    suffix = """
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}"""

    agent1 = initialize_agent(
        tools=[qa_tool],
        llm=llm1,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix.format(few_shot_examples=few_shot_examples),
            "suffix": suffix,
        },
        verbose=True,
        handle_parsing_errors=True,  # Enable error handling
    )



    openai.api_request_timeout = 60  # 将超时时间设置为60秒

    st.markdown("""
    <style>
    /* 固定输入框在底部 */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }

    /* 消息容器留出底部空间 */
    .message-container {
        padding-bottom: 200px;  /* 根据输入框高度调整 */
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    # 初始化会话状态（精简版）
    # if 'generated' not in st.session_state:
    #     st.session_state.generated = []
    #
    # if 'past' not in st.session_state:
    #     st.session_state.past = []

    if 'first_run' not in st.session_state:
        st.session_state.first_run = True

    # 容器定义
    response_container = st.container()  # 响应展示区
    with response_container:
        response_container.markdown('<div class="message-container">', unsafe_allow_html=True)
        # 这里会显示历史消息
        response_container.markdown('</div>', unsafe_allow_html=True)

    input_container = st.container()  # 输入区

    # 欢迎消息（仅首次显示）
    with response_container:
        if st.session_state.first_run:
            st.markdown(f"## Hello! Ask me anything about {uploaded_file.name} 🤗")
            st.session_state.first_run = False



    def process_user_input(query, agent1, container, dfs):

        # 记录用户输入
#        st.session_state.past.append(query)

        # 立即显示用户消息
        with container:
#            message(query, is_user=True, key=f"user_{len(st.session_state.past)}")
            message(query, is_user=True)

            with st.spinner(f'I would first analyze the problem and see which category it belongs to. Let me analyze...'):
                category_original=agent1.invoke(f"What is the problem type in operation of the text? text:{query}")

                def extract_problem_type(output_text):
                    # 定义正则表达式模式，匹配问题类型
                    pattern = r'(Network Revenue Management|Resource Allocation|Transportation|其他问题类型)'
                    match = re.search(pattern, output_text, re.IGNORECASE)
                    return match.group(0) if match else None

                selected_problem = extract_problem_type(category_original['output'])


            with st.spinner(f'I think this is a {selected_problem} Problem. Let me analyze...'):
                # 根据用户的选择，执行相应的代码
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
                       x_1+x_2+x_3+x_4 >=100
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

                    informationC = pd.read_csv('RA_example/Capacity.csv')
                    informationP = pd.read_csv('RA_example/Products.csv')
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

                    with open('RA_example/Sony.txt', 'r', encoding='utf-8') as file:
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
                        for i, r in df.iterrows():
                            description = ""
                            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
                            example_data_description += description + "\n"

                    # 构建 problem_description 和 label

                    with open('TP_example/transportation_problem2.txt', 'r', encoding='utf-8') as file:
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

                result = agent2.invoke(query)
                ai_response = result['output']

            # 记录并显示AI响应
    #        st.session_state.generated.append(ai_response)
        with container:
#            message(ai_response, key=f"ai_{len(st.session_state.generated)}")
            message(ai_response)

        # except Exception as e:
        #     error_msg = f"⚠️ Error processing request: {str(e)}"
        #     st.session_state.generated.append(error_msg)
        #     with container:
        #         message(error_msg, key=f"error_{len(st.session_state.generated)}")


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



    # 用户输入处理
    with input_container:
        input_container.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)

        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="What optimization problem do you want to ask? Please type in the Chatbox in detail",
                height=150,
                key='input'
            )
            submit_button = st.form_submit_button(label='Send')

        input_container.markdown('</div>', unsafe_allow_html=True)

    if submit_button and user_input:
        process_user_input(user_input, agent1, response_container, dfs)

    # 历史消息渲染
#    render_chat_history(response_container)








