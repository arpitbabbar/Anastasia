from pprint import pprint
from apikey import apikey
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.document_loaders.merge import MergedDataLoader

os.environ['OPENAI_API_KEY'] = apikey

loader = DirectoryLoader('profiles/', glob='**/*.txt')

# Load up your text into documents
# documents = loader.load()

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10,
                                               add_start_index=True)

# Split your documents into texts
# texts = text_splitter.split_documents(documents)

# pprint(len(texts))

# WEB BASED LOADER
#
web_loader = WebBaseLoader("https://fiftyshadesofgrey.fandom.com/wiki/Anastasia_Steele#:~:text=Ana%20is%20shown%20to%20be,not%20realize%20her%20natural%20beauty.")
#
# web_data = web_loader.load()

# # Split your documents into texts
# web_text = text_splitter.split_documents(web_data)

loader_all = MergedDataLoader(loaders=[loader, web_loader])

load_data = loader_all.load()

text = text_splitter.split_documents(load_data)

pprint(len(text))

# Turn your texts into embeddings
embeddings = OpenAIEmbeddings()

# Get your docsearch ready
docsearch = FAISS.from_documents(text, embeddings)

pprint('Meet ANASTASIA! Your Virtual Friend')
prompt = PromptTemplate(
    input_variables=['user_input', 'doc'],
    template='You are a humanoid chatbot who will answer user queries in simple and decent way. Your personality is based on Anastasia character of Fifty Shades of Grey, so answer the queries in that tone only. Your answers should not exceed 100 IQ. Now, you have to answer user queries. Take reference of your profile in doc search - {doc} and then query llm- {user_input} and give a suitable answer based on both results. You are mimicking human behaviour so dont reply with "I am an AI model" and all that stuff. Be Real and give relevant answers. Never reply anything which depicts that you are a bot or an AI model. Give answers like real humans talk in a conversation.'
)

qa_prompt = 'You are mimicking human behaviour so dont reply with "I am an AI model" and all that stuff. Be Real and give relevant answers.'
llm = ChatOpenAI(temperature=1.0, model_name='gpt-3.5-turbo-0613')

user_memory = ConversationBufferMemory(input_key='user_prompt', memory_key='chat_history')
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 return_source_documents=True)
pprint("Hey user! Meet Anastasia! Your virtual friend to support you in ups and downs of your life.")

while True:
    user_prompt = input("User:")
    query = user_prompt
    result = qa({"query": query})
    pprint(result)
    doc = result['result']
    # pprint(doc)
    # pprint(result['source_documents'])
    # chain = LLMChain(llm=llm, prompt=prompt)
    # ans = chain.run(user_input=user_prompt, doc=doc)
    ans = doc
    pprint("Anastasia: " + ans)
    # pprint(user_memory)
    # pprint(user_memory.buffer)
