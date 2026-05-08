from store_index import doc_search
from groq import Groq
from Prompt import system_prompt
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq client
llm = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Retriever
retriever = doc_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def get_answer(query):

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Combine retrieved context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Final prompt
    final_prompt = f"""
    {system_prompt}

    Context:
    {context}

    User Question:
    {query}
    """

    # Generate response using Groq
    response = llm.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return answer
