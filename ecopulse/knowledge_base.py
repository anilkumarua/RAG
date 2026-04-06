from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ecopulse.config import AppConfig
from ecopulse.storage import ensure_directory


class HashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            index = hash(token) % self.dimensions
            vector[index] += 1.0
        scale = float(len(tokens))
        return [value / scale for value in vector]


class KnowledgeBase:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embeddings = HashEmbeddings()
        ensure_directory(config.chroma_dir)
        self.vectorstore = Chroma(
            collection_name="eco_pulse_knowledge",
            persist_directory=str(config.chroma_dir),
            embedding_function=self.embeddings,
        )

    def ensure_index(self) -> None:
        if self.vectorstore.get()["ids"]:
            return
        docs = self._load_documents(self.config.knowledge_dir)
        if docs:
            self.vectorstore.add_documents(docs)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def _load_documents(self, knowledge_dir: Path) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        documents: list[Document] = []
        for path in sorted(knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": path.name}))
        return documents
