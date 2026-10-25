from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db_test_step11"
COLLECTION_NAME = "erp_support_test"


def main():
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name}
                )
            )

    if not documents:
        print("No non-empty .txt documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Delete and rebuild only the isolated test collection.
    Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    ).delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Test database: {DB_DIR}")
    print(f"Test collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()