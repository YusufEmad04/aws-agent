from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings
import os
# from dotenv import load_dotenv
from enum import Enum
# from langchain.agents.react.base import DocstoreExplorer
# from langchain.agents import initialize_agent, Tool

# yusuf.emad.pinecone email
pinecone.init(api_key="c869cafc-6f9a-4abf-b8ca-d24ebc2f6ccd", environment="us-west1-gcp-free")

def syllabus_vectorstore():
    return Pinecone.from_existing_index(index_name="agent", embedding=OpenAIEmbeddings())
