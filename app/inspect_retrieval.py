
from pathlib import Path
from app.test_new_db_answers import db, TOP_K, THRESHOLD

question = "What if attendance has not synchronized?"

print("RETRIEVAL DIAGNOSTIC")
print(f"Question: {question}")
print(f"Threshold: {THRESHOLD}")
print(f"Top K: {TOP_K}")
print("-" * 60)

results = db.similarity_search_with_relevance_scores(
    question,
    k=TOP_K,
)

for rank, (doc, score) in enumerate(results, start=1):
    source = Path(
        str(doc.metadata.get("source", "unknown"))
    ).name

    print(f"\nRank: {rank}")
    print(f"Source: {source}")
    print(f"Score: {score:.3f}")
    print(
        "Passes threshold: "
        + ("YES" if score >= THRESHOLD else "NO")
    )
    print("Content:")
    print(doc.page_content[:500])

print("\nDiagnostic complete. No database changes made.")
