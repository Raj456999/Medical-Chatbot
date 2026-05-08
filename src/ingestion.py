from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helper import load_pdf, filter_text, download_embeddings, text_to_chunks
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = 'medical-ai-assistant'

# Load + Process Docs
extracted_data = load_pdf("Medical-Chatbot\Data")
clean_data = filter_text(extracted_data)
chunks = text_to_chunks(clean_data)

# Embeddings
embeddings = download_embeddings()

# Create Index if not exists
if INDEX_NAME not in pc.list_indexes().names():

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Store vectors ONLY ONCE
doc_search = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Vectors Stored Successfully")
