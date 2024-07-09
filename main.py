import streamlit as st
from langchain import hub
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import os

# Set environment variables for API keys
os.environ["GROQ_API_KEY"] = "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"
file_path = os.path.join(os.getcwd(), "ai4i2020.csv")
prompt = """Use the given csv file appropriately to generate explanation for the user query. 
    Create table for the result that you are generating to make it clear for the user.
        Here is the query:"""

# Initialize the LLM
llm = ChatGroq(model="llama3-70b-8192", groq_api_key="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")

# Create a LangChain agent for the CSV file
try:
    agent = create_csv_agent(llm, file_path,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True)

    st.title("Ask your CSV")
    user_input = st.text_input("Ask a question about your CSV:", "")

    if user_input and st.button("Submit"):
        full_input = f"{prompt} {user_input}"
        response = agent.invoke(full_input)
        st.write(response["output"])

except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
except Exception as e:
    st.error(f"Error loading file: {str(e)}")
