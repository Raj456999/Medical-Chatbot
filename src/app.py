from flask import Flask,render_template,jsonify,request
from helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from Prompt import *
import os


app=Flask(__name__)

@app.route("/")
def index():
    return render_template(r'C:\Users\palla\OneDrive\Desktop\Medical chatbot\Medical-Chatbot\templates\inteface.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000, debug=True)