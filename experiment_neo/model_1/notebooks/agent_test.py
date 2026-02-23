from dotenv import load_dotenv
from pydantic import BaseModel
# from langchain_antrhopic import ChatAntropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor


load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatAntropic(model="claude-2", temperature=0)
resposne = llm.invoke("tell me about yourself")
print(resposne)


# setting up promt template
class ResearchResponse(BaseModel):
    # we can make llm calls and make it as complicated as we want
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]



parser=PydanticOutputParser(pydantic_object=ResearchResponse)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)



# setup a prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a research assistant that helps with research tasks. Answer the user query and use necessary tools. Wrap the output in this format and provide no toher text\n{format_instructions}"
         ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())



agent = create_tool_calling_agent(prompt=prompt, llm=llm, tool=[])
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[], verbose=True) 
raw_response = agent_executor.invoke({"query": "what is tactile data?"})
print(raw_response)