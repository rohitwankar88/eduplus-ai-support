
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_name="erp_support"
)

QUESTIONS = [
    "Where can a student get a fee payment receipt?",
    "How can a student troubleshoot a login problem?",
    "How can a student correct an attendance record?",
    "How long does library account activation take?",
    "What is the student's current fee payment status?",
]

THRESHOLD = 0.45

for question in QUESTIONS:
    print("\n" + "=" * 65)
    print("Question:", question)

    results = db.similarity_search_with_relevance_scores(
        question,
        k=5,
    )

    if not results:
        print("No documents retrieved.")
        continue

    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown")
        status = "KEEP" if score >= THRESHOLD else "BELOW THRESHOLD"

        print(
            f"\n{rank}. Source: {source}"
            f"\n   Score: {score:.3f} | {status}"
            f"\n   Preview: {' '.join(doc.page_content.split())[:180]}"
        )
