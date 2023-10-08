import pickle
from langchain.vectorstores import Pinecone
import hashlib
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from classes import SelfQueryRetrieverNew
from langchain.vectorstores.base import VectorStoreRetriever
import os
# from dotenv import load_dotenv
from enum import Enum
# from langchain.agents.react.base import DocstoreExplorer
# from langchain.agents import initialize_agent, Tool

# yusuf.emad.pinecone email
pinecone.init(api_key=os.environ["PINECONE_API_KEY4"], environment=os.environ["PINECONE4_ENV"])


class VectorStoreType(str, Enum):
    DIAMONDS = "diamonds"
    LAW_FIRM = "law_firm"
    BEAUTY_CLINIC = "beauty_clinic"
    CRYPTO = "crypto"
    JEWELRY = "jewelry"
    SYLLABUS = "syllabus"

def syllabus_vectorstore():
    return Pinecone.from_existing_index(index_name="mvp-agent", embedding=OpenAIEmbeddings())
