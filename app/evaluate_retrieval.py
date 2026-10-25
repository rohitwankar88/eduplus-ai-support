
from pathlib import Path
from app.test_new_db_answers import (
    db,
    TOP_K,
    THRESHOLD,
    MAX_SCORE_GAP,
)

# Retrieval evaluation only. Does not modify either database.

test_cases = [
    {
        "question": "How do I troubleshoot login problems?",
        "expected_source": "login_support.txt",
    },
    {
        "question": "I cannot sign in to the ERP portal.",
        "expected_source": "login_support.txt",
    },
    {
        "question": "I forgot my password and cannot log in.",
        "expected_source": "login_support.txt",
    },
    {
        "question": "Where can I download my fee receipt?",
        "expected_source": "fee_support.txt",
    },
    {
        "question": "My payment was made but no receipt appeared.",
        "expected_source": "fee_support.txt",
    },
    {
        "question": "How should staff troubleshoot a missing attendance record?",
        "expected_source": "attendance_support.txt",
    },
    {
        "question": "What if attendance has not synchronized?",
        "expected_source": "attendance_support.txt",
    },
    {
        "question": "How long does library account activation take?",
        "expected_source": "library_support.txt",
    },
    {
        "question": "How do I activate my library access?",
        "expected_source": "library_support.txt",
    },
]

passed = 0
failed = 0

print("\nRAG RETRIEVAL EVALUATION")
print(f"Test database: {Path('chroma_db_test_step11').resolve()}")
print(f"Top K: {TOP_K}")
print(f"Relevance threshold: {THRESHOLD}")
print(f"Maximum score gap: {MAX_SCORE_GAP}")
print("=" * 65)

for index, case in enumerate(test_cases, start=1):
    question = case["question"]
    expected = case["expected_source"]

    try:
        results = db.similarity_search_with_relevance_scores(
            question,
            k=TOP_K,
        )

        if results:
            best_score = results[0][1]

            dynamic_threshold = max(
                THRESHOLD,
                best_score - MAX_SCORE_GAP,
            )

            qualifying = [
                (doc, score)
                for doc, score in results
                if score >= dynamic_threshold
            ]
        else:
            qualifying = []

        sources = []

        for doc, score in qualifying:
            source = Path(
                str(doc.metadata.get("source", "unknown"))
            ).name

            if source not in sources:
                sources.append(source)

        success = expected in sources

        if success:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {index}: {question}")
        print(f"Expected source: {expected}")
        print(f"Retrieved sources: {sources}")
        print(f"Result: {'PASS' if success else 'FAIL'}")

        if qualifying:
            print("Retrieved chunks:")

            for doc, score in qualifying:
                source = Path(
                    str(doc.metadata.get("source", "unknown"))
                ).name

                preview = doc.page_content.replace("\n", " ")[:160]

                print(f"  - {source} | score={score:.3f}")
                print(f"    {preview}...")
        else:
            print("No chunks passed the relevance threshold.")

    except Exception as error:
        failed += 1
        print(f"\nTest {index}: {question}")
        print("Result: ERROR")
        print(f"Details: {error}")

total = passed + failed
pass_rate = passed / total * 100 if total else 0

print("\n" + "=" * 65)
print(f"Total tests: {total}")
print(f"Passed: {passed}")
print(f"Failed/errors: {failed}")
print(f"Expected-source retrieval pass rate: {pass_rate:.1f}%")
print("=" * 65)
print("No database modifications were performed.")
