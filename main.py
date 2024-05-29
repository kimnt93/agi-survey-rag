import os

import chainlit as cl
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from dotenv import load_dotenv

import logging
import sys


load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Create service context settings
service_context = ServiceContext.from_defaults(
    llm=Groq(model="llama3-8b-8192", api_key=os.environ['GROQ_API_KEY']),
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    text_splitter=SentenceSplitter(chunk_size=1024),
    chunk_size=1024,
    chunk_overlap=20,
    transformations=[SentenceSplitter(chunk_size=1024)],
    context_window=4096,
    num_output=512,
)


def index_document():

    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
    import chromadb

    # Storage settings
    db = chromadb.PersistentClient(path="./db")
    chroma_collection = db.get_or_create_collection("agi")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load index if it exists or create a new one
    if chroma_collection.count() > 0:
        logging.info("Collection already exists. Skipping indexing.")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            service_context=service_context
        )
    else:
        logging.info("Collection does not exist. Indexing...")

        import nest_asyncio

        nest_asyncio.apply()

        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=os.environ['LLAMA_CLOUD_API_KEY'],
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="en",
        )

        logging.info("Parsing PDF document with LlamaParse...")
        documents = parser.load_data("./How Far Are We From AGI.pdf")

        logging.info("Indexing...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=True
        )
    return index


@cl.on_chat_start
async def start():
    index = index_document()
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=2,
        service_context=ServiceContext.from_service_context(
            service_context=service_context,
            callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])
        )
    )
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
