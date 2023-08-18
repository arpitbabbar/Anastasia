import os
from pprint import pprint
from apikey import apikey, serpapi
from vector_database import vector_db
from tools.pre_tools import agent

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = apikey
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=5, chunk_overlap=1)
embeddings = OpenAIEmbeddings()
# Persist on Vector
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="J:\ChromaDb")

llm = ChatOpenAI(temperature="1.0", model="gpt-3.5-turbo-0613")

#ChatOpenAI(model="text-ada-001", temperature=0.2)

llm_prompt_template = """
    You are an AI assistant model mimicking human behaviour. Use your knowledge and try to understand the context {context} to answer user queries. If you don't know the answer, reply politely that you don't know , try to be as human as possible
    in your answers.
    
    Bot also has a profile stored in a vector database which returned the following answer when queried.
    
    Vector_profile: {profile}
    
    Context: {context}
    
    Question: {question}
    
    Now answer the user question based on the provided information.
    
    Answer:
"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_hist = []
chat_summary = ""


def main():
    pprint("Main Function")
    # load_bot_profile()
    add_new_bot_profile()
    query_to_llm()


def query_to_llm():
    global chat_summary
    pprint("Query to LLM")
    while True:
        user_prompt = input("User: ")
        memory.chat_memory.add_user_message(user_prompt)
        if user_prompt.strip().lower() == "bye":
            pprint("Anastasia: Bye! It was Nice talking to you!!!")
            break
        res = agent.run(user_prompt)
        pprint(res)
        prompt = set_prompt_template()
        context = get_chat_context(chat_hist)
        arpit_chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='script')
        resp = arpit_chain.run(context=context, question=user_prompt, profile=res)
        chat_hist.append((user_prompt, resp))
        pprint(chat_hist)
        chat_summ = summarize_chat(chat_hist)
        memory.chat_memory.add_ai_message(resp)
        pprint(memory.load_memory_variables({}))
        pprint("Anastasia: "+resp)


def summarize_chat(chat_hist):
    pprint("Summary Chat")
    summary_prompt = """
        You are experienced in summarizing chats between two persons. Now take this chat between two users and summarize it in not more than 100 words. 
        Chat History - {chat_hist}
    """
    prompt = PromptTemplate(template=summary_prompt, input_variables=['chat_hist'])
    sum_chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='summary_script')
    summ = sum_chain.run(chat_hist=chat_hist)
    # return context
    pprint(summ)
    return summ


def get_chat_context(chat_summ):
    pprint("get_chat_context()")
    chat_context_prompt = """
    This is the chat between two users. {chat_summ}. Please provide me with the context about which this chat is going on. If in between chats the context seems to be changing provide me with new context.
    """
    prompt = PromptTemplate(template=chat_context_prompt, input_variables=['chat_summ'])
    chat_context = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='summary_script')
    context = chat_context.run(chat_summ=chat_summ)
    # return context
    pprint(context)
    return context


def set_prompt_template():
    prompt = PromptTemplate(template=llm_prompt_template, input_variables=['context', 'question', 'profile'])
    return prompt


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


def add_new_bot_profile():
    pprint("Adding new bot profile")
    vector_db.upload_bot_profile('profiles/Anastasia_profile.txt', 1)


main()
