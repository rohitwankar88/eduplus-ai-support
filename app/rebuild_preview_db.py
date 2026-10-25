
from pathlib import Path
import re
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_DB_DIR = BASE_DIR / "chroma_db_test_step11"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def split_fee_document(text):
    header, body = text.split("Issue 1:", 1)
    issue1, remaining = body.split("Issue 2:", 1)

    section1 = (
        header.strip()
        + "\n\nIssue 1: "
        + issue1.strip()
    )

    rest = "Issue 2:" + remaining

    sections = [section1]
    sections.extend(
        part.strip()
        for part in re.split(
            r"(?=Safety rules:|Note:)", rest
        )
        if part.strip()
    )

    return sections


def main():
    documents = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if path.name == "fee_support.txt":
            sections = split_fee_document(text)

            for section in sections:
                for chunk in splitter.split_text(section):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": path.name}
                        )
                    )
        else:
            for chunk in splitter.split_text(text):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": path.name}
                    )
                )

    if TEST_DB_DIR.exists():
        shutil.rmtree(TEST_DB_DIR)

    db = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(
            model="nomic-embed-text"
        ),
        persist_directory=str(TEST_DB_DIR),
        collection_name="erp_support_test"
    )

    print("Test database created successfully.")
    print("Proposed chunks:", len(documents))
    print("Stored chunks:", db._collection.count())
    print("Test database folder:", TEST_DB_DIR)


if __name__ == "__main__":
    main()
