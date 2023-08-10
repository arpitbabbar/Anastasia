import os
from pprint import pprint
from apikey import apikey, serpapi

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


os.environ['OPENAI_API_KEY'] = apikey
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=5, chunk_overlap=1)
embeddings = OpenAIEmbeddings()
# Persist on Vector
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="J:\ChromaDb")


def main():
    pprint("Main Function")
    load_bot_profile()


def query_to_llm(query):
    pprint("Query to LLM")


def load_bot_profile():
    pprint("Loading Bot Profile")
    # loader = DirectoryLoader('profiles/', glob='**/*.txt')
    loader = TextLoader('profiles/Anastasia_profile.txt')
    document = loader.load()
    # pprint(len(document))
    # pprint(document)
    load_docs_to_vector(document)


def load_docs_to_vector(document):
    pprint("Loading documents to vector")
    texts = text_splitter.split_documents(document)
    pprint(len(texts))
    pprint(texts)
    db = Chroma.from_documents(texts, OpenAIEmbeddings())


main()