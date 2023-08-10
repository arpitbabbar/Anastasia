import os
from apikey import apikey, serpapi

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10,add_start_index=True)
embeddings = OpenAIEmbeddings()
# Persist on Vector
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="J:\ChromaDb")


def main():
    print("Main Function")
    os.environ['OPENAI_API_KEY'] = apikey


def query_to_llm(query):
    print("Query to LLM")


def load_bot_profile():
    print("Loading Bot Profile")
    loader = DirectoryLoader('profiles/', glob='**/*.txt')
    document = loader.load()
    print(document)


def load_docs_to_vector(document):
    print("Loading documents to vector")
    db = Chroma.from_documents(document, OpenAIEmbeddings())