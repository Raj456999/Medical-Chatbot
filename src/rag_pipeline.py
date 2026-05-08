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

    try:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = f"""
        {system_prompt}

        Context:
        {context}

        User Question:
        {query}
        """

        response = llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.3,
            max_tokens=512
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:
        return f"ERROR: {str(e)}"
