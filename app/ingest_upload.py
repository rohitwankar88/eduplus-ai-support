
from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.document_processor import process_uploaded_file

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def get_database():
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="erp_support"
    )


def index_uploaded_file(uploaded_file):
    filename = Path(uploaded_file.name).name
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    db = get_database()

    # Prevent duplicate indexing for files with stored hashes
    existing = db.get(include=["metadatas"])

    for metadata in existing["metadatas"]:
        if metadata.get("file_hash") == file_hash:
            raise ValueError(
                f"'{filename}' has already been indexed."
            )

    # Process and index the uploaded file
    chunks = process_uploaded_file(uploaded_file)

    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash

    db.add_documents(chunks)

    return len(chunks)


def get_indexed_documents():
    db = get_database()
    result = db.get(include=["metadatas"])

    documents = set()

    for metadata in result["metadatas"]:
        filename = metadata.get("source")

        if filename:
            documents.add(Path(filename).name)

    return sorted(documents)
