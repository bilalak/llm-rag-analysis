import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from langchain_core.documents import Document


from dotenv import load_dotenv
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")

if __name__ == "__main__":
    file_path='./mom.json'
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    # print(data)
    # data = "This is an example page content."
    # pages = [Document(
    #                 page_content=json.dumps(data, indent=0),
    #                 metadata={"source": "./data.json"},
    #             )]
    
    splitter = RecursiveJsonSplitter(max_chunk_size=1024)

    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # embeddings = embeddings_model.embed_documents(texts=csv_data)
    
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
        splitter,
        embeddings_model,
        index_name=index_name,
        namespace=namespace,
    )