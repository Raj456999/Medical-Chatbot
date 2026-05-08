from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

# To load pdf and return the content in pdf
def load_pdf(Data):
    loader=DirectoryLoader(
        Data,glob='*.pdf',loader_cls=PyPDFLoader)
    docs=loader.load()
    return docs
#To filter the text for training
def filter_text(docs:List[Document])->List[Document]:
    modified_data: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get('source')
        modified_data.append(Document(
            page_content=doc.page_content,metadata={'source':src}
        ))
    return modified_data

# Splitting documents into smaller chunks
def text_to_chunks(data):
    chunks=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text=chunks.split_documents(data)
    return text
#Downloading embedding model
def download_embeddings():
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings



