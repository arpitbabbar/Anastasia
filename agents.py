import os

import openai
from langchain import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, ImageCaptionLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PIL import Image
import requests
from langchain.indexes import VectorstoreIndexCreator


from apikey import apikey, serpapi

print(apikey)
os.environ['OPENAI_API_KEY'] = apikey
os.environ['SERPAPI_API_KEY'] = serpapi

llm = OpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)


list_image_urls = [
    "https://images.moneycontrol.com/static-mcnews/2023/08/Collage-Maker-05-Aug-2023-08-27-AM-2603-770x433.jpg?impolicy=website&width=770&height=431"
]


query = "Generate a image of Emma Stone"
#
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

docloader = DirectoryLoader('profiles/', glob='**/*.txt')
# Load up your text into documents
documents = docloader.load()

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

doctext = text_splitter.split_documents(documents)
#
# db = Chroma.from_documents(webtexts, OpenAIEmbeddings())
db = Chroma.from_documents(doctext, OpenAIEmbeddings())

# res = vectorstore.similarity_search()
#
# query = "Give brief of the docs"
matching_docs = db.similarity_search(query)
#
# print(len(matching_docs))
# print(matching_docs)

image_loader = ImageCaptionLoader(path_images=list_image_urls)
list_docs = image_loader.load()
print(list_docs)

Image.open(requests.get(list_image_urls[0], stream=True).raw).convert("RGB")

index = VectorstoreIndexCreator().from_loaders([image_loader])

image_query = "Do you know this guy?"
res = index.query(image_query)

print(res)

search = SerpAPIWrapper()


def searchDocs(ques):
    print("Search in DOcs")
    docsearch = db.similarity_search(ques)
    return docsearch

#
# query = "Do you have any boyfriend?"
#
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]
#
# # tools = load_tools(["serpapi", "llm-math"], llm=llm)
#
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# #
# result = agent.run(query)
# #
# print(result)





# WEB BASED LOADER
#
# loader = WebBaseLoader("https://www.nationalarchives.gov.uk/education/resources/hitler-assassination-plan/")
#
# data = loader.load()

# # Split your documents into texts
# webtexts = text_splitter.split_documents(data)






# CHAIN RND

# chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
# answer = chain.run(input_documents=matching_docs, question=query)
# print(answer)
#
#
# def multiply(a, b):
#     return a * b.

# https://python.langchain.com/docs/modules/agents/how_to/agent_vectorstore
# COnversation SUmmary memory
# Conversation chain
# https://www.youtube.com/watch?v=ziu87EXZVUE

# prompt = PromptTemplate(
#     input_variables=["input"],
#     template="Tell Me your views about {input}"
# )
#
# chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
#
# output = chain.run("Oppenheimer")
# print(output)






























# IMAGE GEN USING DALL-E

# gpt_prompt = "Write a description of image which I want to generate such that it looks humanly and I want to feed it to image generation models. Image of Ana De Armas"
#
# resp = llm(gpt_prompt)
# print(resp)
#
# response = openai.Image.create(
#     prompt=resp,
#     size="1024x1024",
#     response_format="url"
# )
#
# print(response["data"][0]["url"])
