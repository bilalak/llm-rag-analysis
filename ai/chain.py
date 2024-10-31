from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv()


def get_ai_response(message: str, session_id: str, store):
    
    print("message", message)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a knowledgeable assistant specializing in window and door manufacturing. Your role is to provide detailed and accurate information about different types of windows and doors, their manufacturing processes, materials used, and best practices in the industry. You should follow up on previously asked questions to maintain a coherent and helpful conversation. If you are unsure about an answer, provide the best possible information you have based on general knowledge.
            If the follow up question does not need context like when the follow up question is a remark like: excellent, thanks, thank you etc., return the exact same text back."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
    )
    
    chain = prompt | llm | StrOutputParser()

    wrapped_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        history_messages_key="chat_history",
    )
    response = wrapped_chain.invoke({"input": message}, config={"configurable": {"session_id": session_id}})
    print('response 123456 =====> ', response)
    
    
    return response
    


    