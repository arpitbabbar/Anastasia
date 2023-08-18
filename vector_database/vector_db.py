import os

from apikey import apikey
from datetime import datetime
from pprint import pprint

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


os.environ['OPENAI_API_KEY'] = apikey

# chroma db detail
chroma_db_base_path = 'J:\ChromaDb'
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=chroma_db_base_path)

# transformer pattern
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=7,
    chunk_overlap=2
)


def upload_bot_profile(file, bot_id):
    print("upload_bot_profile called")
    # loading data
    raw_documents = TextLoader(file).load()
    documents = text_splitter.split_documents(raw_documents)
    documents_with_header = []
    for doc in documents:
        doc.metadata.__setitem__("bot_id", bot_id)
        doc.metadata.pop("source")
        documents_with_header.append(doc)
        print(doc)
    vectorstore.add_documents(documents_with_header)


def upload_bot_profile_dir(dir, bot_id):
    print("upload_bot_profile called")
    # loading data
    loader = DirectoryLoader('../profiles/life/', glob="*.txt")
    pprint(loader)
    raw_documents = loader.load()
    documents = text_splitter.split_documents(raw_documents)
    documents_with_header = []
    for doc in documents:
        doc.metadata.__setitem__("bot_id", bot_id)
        doc.metadata.pop("source")
        documents_with_header.append(doc)
        print(doc)
    vectorstore.add_documents(documents_with_header)


def add_bot_profile(data, bot_id):
    print("add_bot_profile called")
    # loading data
    documents = text_splitter.split_documents(data)
    documents_with_header = []
    for doc in documents:
        doc.metadata.__setitem__("bot_id", bot_id)
        doc.metadata.pop("source")
        documents_with_header.append(doc)
    vectorstore.add_documents(documents_with_header)


def fetch_bot_profile(bot_id, query, count):
    print("fetch_bot_profile called")
    docs = vectorstore.similarity_search(query, k=count, filter={"bot_id": bot_id})
    return docs


def add_user_profile(data, user_id):
    print("add_user_profile called")
    documents = text_splitter.split_documents(data)
    documents_with_header = []
    for doc in documents:
        doc.metadata.__setitem__("user_id", user_id)
        doc.metadata.pop("source")
        documents_with_header.append(doc)
    vectorstore.add_documents(documents_with_header)


def fetch_user_profile(user_id, query, count):
    print("add_user_profile called")
    docs = vectorstore.similarity_search(query, k=count, filter={"user_id": user_id})
    return docs


def add_conversation(data, bot_id, user_id):
    print("add_user_profile called")
    documents = text_splitter.split_documents(data)
    documents_with_header = []
    for doc in documents:
        doc.metadata.__setitem__("bot_id", bot_id)
        doc.metadata.__setitem__("user_id", user_id)
        doc.metadata.__setitem__("time", datetime.now())
        doc.metadata.pop("source")
        documents_with_header.append(doc)
    vectorstore.add_documents(documents_with_header)


def fetch_conversation(bot_id, user_id, query, count):
    print("add_user_profile called")
    docs = vectorstore.similarity_search(query, k=count, filter={"bot_id": bot_id, "user_id": user_id})
    return docs
