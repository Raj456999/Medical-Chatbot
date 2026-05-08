from store_index import doc_search
# from langchain_groq import ChatGroq
import groq
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
llm =  Groq(api_key=os.getenv("GROQ_API_KEY"))
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
    response = llm.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
    answer = response.choices[0].message.content
    return answer
   

