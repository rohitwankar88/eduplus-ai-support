
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db_test_step11"

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=OllamaEmbeddings(
        model="nomic-embed-text"
    ),
    collection_name="erp_support_test"
)

questions = [
    "Where can a student get a fee payment receipt?",
    "How can a student troubleshoot a login problem?",
    "How can a student correct an attendance record?",
    "How long does library account activation take?",
    "What is the student's current fee payment status?",
]

THRESHOLD = 0.50

for question in questions:
    print("\n" + "=" * 65)
    print("Question:", question)

    results = db.similarity_search_with_relevance_scores(
        question,
        k=3
    )

    relevant_count = 0

    for rank, (doc, score) in enumerate(results, start=1):
        is_relevant = score >= THRESHOLD

        if is_relevant:
            relevant_count += 1

        print(
            f"\n{rank}. Source: "
            f"{Path(doc.metadata.get('source', 'Unknown')).name}"
            f"\n   Score: {score:.3f}"
            f"\n   Threshold: "
            f"{'PASS' if is_relevant else 'FILTERED'}"
            f"\n   Preview: "
            f"{' '.join(doc.page_content.split())[:180]}"
        )

    print(
        f"\nDocuments passing threshold: "
        f"{relevant_count}/{len(results)}"
    )

print("\nDiagnostic complete. No database changes made.")
