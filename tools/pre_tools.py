import os
from pprint import pprint
from apikey import apikey, serpapi
from vector_database import vector_db


from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = apikey
os.environ['SERPAPI_API_KEY'] = serpapi
llm = ChatOpenAI(temperature="1.0", model="gpt-3.5-turbo-0613")


def searchDocs(ques):
    print("Search in DOcs")
    docsearch = vector_db.fetch_bot_profile(1, ques, 5)
    return docsearch


search = SerpAPIWrapper()

#
# query = "Do you have any boyfriend?"
#
# LLM Search -  Generate Info and persist
# Tool for user, if user telling info about themself
tools = [
    Tool(
        name="Profile Search",
        func=searchDocs,
        description="Used when a question is asked about the profile of the bot with which user is having a conversation.",
    )
]

agent = initialize_agent(
    tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)