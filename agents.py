from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.agent import AgentExecutor
from classes import DynamoDBChatMessageHistoryNew
from retrievers import *
from tools import get_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
# from dotenv import load_dotenv
from tools import AgentTool

def _init_test_agent(session_id):
    llm_chat = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-16k-0613")
    tools = [
        get_tool("retriever")(
            syllabus_vectorstore(),
            name="syllabus_database",
            description="retrieve manufacturing techology syllabus data"
        ),
        # get_tool(AgentTool.TELEGRAM)(
        #     description="used to send a message to the teacher in case the user wanted a human to answer him."
        # )
    ]

    sys_message = SystemMessage(
        content="You are a manufacturing techology teacher. You help students in their questions about the syllabus.\n"
                "You always lookup question in the syllabus_database with the same user question.\n"
                "You NEVER answer questions outside the syllabus, and never come up with answers.\n"
                "You always try to refer to the syllabus and in which lesson or topic the answer is if you can.\n"
                "Query the syllabus_database with the exact user message without any modifications.\n"
                "Be helpful and always willing to answer more and more questions.\n"
                "Begin the conversation with offering help in manufacturing techology questions.\n"

    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=sys_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    reminder = "NEVER come up with answers. Always refer to the syllabus and in which lesson the answer is whenever possible."

    memory = AgentTokenBufferMemory(max_token_limit=10500, memory_key="chat_history", llm=llm_chat,
                                    chat_memory=DynamoDBChatMessageHistoryNew(table_name="langchain-agents",
                                                                              session_id=session_id, reminder=reminder))

    agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor
# enum Agent
class Agent(str, Enum):
    JEWELRY = "jewelry"
    BIZNIS_CLINICS = "biznis-clinics"
    CRYPTO = "crypto"
    ECOM = "ecom"
    BEAUTY_CLINICS = "beauty-clinics"
    DIAMONDS = "diamonds"
    TEST = "test"
    WELCH_LAW = "welch-law"


agents_dict = {
    Agent.TEST: _init_test_agent,
}

def get_agent(name, session_id):
    return agents_dict[name](session_id=session_id)
