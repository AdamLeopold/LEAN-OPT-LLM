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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['LANGCHAIN_PROJECT'] = "or_expert"

user_api_key = st.sidebar.text_input(
    label="#### Your langchain API key 👇",
    placeholder="Paste your langchain API key, sk-",
    type="password")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name


    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()

    embeddings = OllamaEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

    llm = Ollama(model='llama3')

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-4', openai_api_key=user_api_key),
        retriever=vectors.as_retriever())

    agent = create_csv_agent(llm, tmp_file_path, verbose=True, allow_dangerous_code=True)
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

# streamlit run tuto_chatbot_csv.py