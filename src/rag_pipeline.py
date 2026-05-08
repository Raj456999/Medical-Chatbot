from store_index import doc_search
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from Prompt import system_prompt
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY=os.getenv('GROQ_API_KEY')


retrieved_doc=doc_search.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3}
)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
rag_chain = (
    {
        "context": retrieved_doc,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)
def get_answer(query):
    response = rag_chain.invoke(query)
    return response.content

