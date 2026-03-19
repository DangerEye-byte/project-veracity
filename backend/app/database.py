# app/database.py
from langchain_qdrant import QdrantVectorStore  # The modern class
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None
bm25_retriever = None
ensemble_retriever = None

def initialize_retriever(docs):
    global vectorstore, bm25_retriever, ensemble_retriever
    
    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name="veracity_vault",
        force_recreate=True
    )
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever()],
        weights=[0.3, 0.7]
    )
    return ensemble_retriever

def get_vectorstore():
    return vectorstore