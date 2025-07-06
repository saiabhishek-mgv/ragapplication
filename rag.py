from uuid import uuid4
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()
GROQ_API = os.getenv("GROQ_API")
#constants
CHUNK_SIZE = 100
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTOR_STORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

llm = None
vector_store = None

def initialize_components():
    """This function initializes the components required for the RAG system.
    :return: None
    """
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.9, max_tokens=500, api_key=GROQ_API)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef
            # no persist_directory
        )



def process_urls(urls):
    """This function scrapes data from url and store it in a vector store.
    :param urls: List of URLs to scrape
    :return: None
    """
    yield "Initializing components..."
    #vector_store.delete_collection()
    initialize_components()

    yield "Loaded data"
    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    data = loader.load()
    yield "split text"
    
    text_splitter = RecursiveCharacterTextSplitter(
        #separator=["\n\n","\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0
    )
    docs = text_splitter.split_documents(data)

    yield "Add Docs to Vector Store" 
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding documents to vector store"

    # chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever()
    # )
    # return chain

def generate_answer(query):
    """This function generates an answer to the question using the RAG system.
    :param question: The question to answer
    :return: The answer to the question
    """
    if not vector_store:
        raise ValueError("Vector store is not initialized.")

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get('sources', "")
    return result['answer'], sources

if __name__ == "__main__":
    urls = [
        'https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html',
        'https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html'
    ]

    process_urls(urls)
    answer, sources = generate_answer("What was the 30-year fixed mortgage rate?")
    print (f"Answer: {answer}")
    print (f"Sources: {sources}")