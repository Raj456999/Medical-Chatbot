from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from helper import download_embeddings
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

INDEX_NAME = 'medical-ai-assistant'

# Download embeddings
embeddings = download_embeddings()

# Load existing vector DB ONLY
doc_search = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)