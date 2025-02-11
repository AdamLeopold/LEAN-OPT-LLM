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



user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 在侧边栏添加选择框
    problem_options = [
        "Network Revenue Management",
        "Uncapacitated Facility Location Problem",
        "Knapsack Problem",
        "Transportation Problem",
        "Minimum-Cost Flow Problem",
        "Assignment Problem",
        "Capacitated Facility Location Problem"
    ]

    selected_problem = st.sidebar.selectbox("请选择一个问题", problem_options)

    # 根据用户的选择，执行相应的代码
    if selected_problem == "Network Revenue Management":

        information = pd.read_csv('nike Shoes Sales.csv')
        information_head = information[:36]

        # 将示例数据转换为字符串，供 few_shot_examples 使用
        example_data_description = "\nHere is the product data:\n"
        for index, row in information_head.iterrows():
            example_data_description += f"Product {index + 1}: {row['Product Name']}, revenue w_{index + 1} = {row['Revenue']}, demand rate a_{index + 1} = {row['Demand']}, initial inventory c_{index + 1} = {row['Initial Inventory']}\n"

        # 构建 problem_description 和 label
        problem_description0 = problem_description = """The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer."""

        problem_description += example_data_description

        problem_description1 = 'The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories of the Nike x Olivia Kim brand, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer.'

        label_head1 = """
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
        df = pd.read_csv(tmp_file_path)

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
        # user_description = 'The data of the store offers several styles of Nike shoes is provided in "Nike Shoes Sales.csv". Through the dataset, the revenue of each shoes is listed in the column "revenue". The demand for each style is independent. The store’s objective is to maximize the total expected revenue based on the fixed initial inventories of the Kyrie brand, which are detailed in column “inventory.” During the sales horizon, no replenishment is allowed, and there are no in-transit inventories. Customer arrivals, corresponding to demand for different styles of Nike shoes, occur in each period according to a Poisson process, with arrival rates specified in column “demand.” The decision variables y_i represent the number of customer requests the store intends to fulfill for Nike shoe style i, with each y_i being a positive integer.'

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
            st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # container for the chat history
        response_container = st.container()
        # container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# streamlit run gptrag_2.py