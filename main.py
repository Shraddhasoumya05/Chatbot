import streamlit as st
from langchain_groq import ChatGroq
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = "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"

from langchain_groq import ChatGroq

llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key="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")


# Initialize the loader
loader = TextLoader("/Users/shraddhasoumya/Downloads/ai4i2020.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatGroq(model="mixtral-8x7b-32768")
    | StrOutputParser()
)





st.title("Ask your CSV")
user_input = st.text_input("Ask a question about your CSV:", "")
prompt = """
        You must need to use matplotlib library if required to create any chart.

        If the query requires creating a table, make the table in markdown format.

        You must also try to visualise the table with appropriate plots. Make the plot in markdown format.
        
        If the query is just not asking for a chart, but requires a response, reply the answer in simple markdown.

        
        Lets think step by step.

        Here is the query:
        ---""" + f"""
        {user_input}
        ---
    )"""


if st.button("Submit"):
    response = rag_chain.invoke(prompt)
    st.write(response)