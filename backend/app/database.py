from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List

# ---------------------------------------------------------------------------
# Shared model instance — imported by auditor.py, not reloaded
# ---------------------------------------------------------------------------
shared_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Thin LangChain-compatible wrapper around the shared ST model
# LangChain needs an object with .embed_documents() and .embed_query()
# ---------------------------------------------------------------------------
class SharedSTEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return shared_embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return shared_embedding_model.encode(text, convert_to_numpy=True).tolist()

embeddings = SharedSTEmbeddings()

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
vectorstore       = None
bm25_retriever    = None
ensemble_retriever = None

def initialize_retriever(docs):
    global vectorstore, bm25_retriever, ensemble_retriever

    client = QdrantClient(url="http://localhost:6333")

    # Only recreate collection if it doesn't already exist
    existing = [c.name for c in client.get_collections().collections]
    if "veracity_vault" in existing:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="veracity_vault",
            embedding=embeddings,
        )
    else:
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            embeddings,
            url="http://localhost:6333",
            collection_name="veracity_vault",
        )

    bm25_retriever   = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever()],
        weights=[0.3, 0.7]
    )
    return ensemble_retriever

def get_vectorstore():
    return vectorstore