import os
from flask import Flask, request, Response, stream_with_context, jsonify
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
import time
from pinecone import Pinecone, ServerlessSpec
import json
from werkzeug.utils import secure_filename

from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
# PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"

# CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, origins="*")


def get_response(message: str):
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL_NAME,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    pc = Pinecone(api_key=PINECONE_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index = pc.describe_index(PINECONE_INDEX)
    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_KEY, index_name=PINECONE_INDEX, embedding=embeddings,
                                      namespace=PINECONE_NAMESPACE)
    retriever = vectorstore.as_retriever()
    # docs = retriever.invoke(message)

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context.

        <context>
        {context}
        </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant "
                "to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            # If only one message, then we just pass that message's content to retriever
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )
    response = conversational_retrieval_chain.invoke(
        {
            "messages": [
                HumanMessage(content=message),
            ],
        }
    )
    
    print('response======>', response["answer"])
                
    return response["answer"]


@app.route("/embed", methods=["POST"])
# @cross_origin()
def embed_json():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # You could validate the file name or its extensions here
        filename = secure_filename(file.filename)
        
        # Read and parse the JSON file
        data = json.load(file)

        documents = []
        for item in data:
            documents.append(Document(
                page_content=str(item),  # Adjust based on what you want from each item
                metadata={},
            ))

        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # embeddings = embeddings_model.embed_documents(texts=csv_data)

        print(PINECONE_KEY)
        pc = Pinecone(api_key=PINECONE_KEY)

        index_name = PINECONE_INDEX
        namespace = PINECONE_NAMESPACE

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        index = pc.Index(index_name)

        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
        docsearch = PineconeVectorStore.from_documents(
            documents,
            embeddings_model,
            index_name=index_name,
            namespace=namespace,
        )

        return "Success"
    else:
        return jsonify({"error": "Unsupported file type"}), 400


@app.route("/chat", methods=["POST"])
# @cross_origin()
def sse_request():
    body = request.json
    message = body.get("message")
    print(message)
    return Response(get_response(message))


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
