from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


search = TavilySearch()


llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(llm, tools)


def main():
    print("Hello from langchain!")
    result = agent.invoke({"messages": [HumanMessage(content="provide 3 job openings for AI engineer using langchain in India with highest salary package")]})
    print(result)


if __name__ == "__main__":
    main()
