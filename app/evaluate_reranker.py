from app.test_new_db_answers import db, rerank_chunks
TEST_CASES = [
    ("How do I troubleshoot login problems?", "login_support.txt"),
    ("I forgot my password and cannot log in.", "login_support.txt"),
    ("Where can I download my fee receipt?", "fee_support.txt"),
    ("My payment was made but no receipt appeared.", "fee_support.txt"),
    ("How should staff troubleshoot a missing attendance record?", "attendance_support.txt"),
    ("How long does library account activation take?", "library_support.txt"),
    ("How do I activate my library access?", "library_support.txt"),
]
TOP_K = 5
def get_source(doc):
    return doc.metadata.get("source", "").replace("\\", "/").split("/")[-1]
def main():
    vector_hits = 0
    reranked_hits = 0
    print("\nRERANKING BEFORE-AND-AFTER EVALUATION\n")
    for question, expected_source in TEST_CASES:
        results = db.similarity_search_with_relevance_scores(
            question, k=TOP_K
        )
        before = [doc for doc, score in results]
        after = rerank_chunks(question, results)
        before_sources = [get_source(doc) for doc in before]
        after_sources = [get_source(item[0]) for item in after]
        before_ok = expected_source in before_sources[:3]
        after_ok = expected_source in after_sources[:3]
        vector_hits += int(before_ok)
        reranked_hits += int(after_ok)
        print(f"Question: {question}")
        print(f"Expected: {expected_source}")
        print(f"Vector top 3: {before_sources[:3]}")
        print(f"Reranked top 3: {after_sources[:3]}")
        print(f"Vector correct in top 3: {before_ok}")
        print(f"Reranked correct in top 3: {after_ok}")
        print("-" * 60)
    total = len(TEST_CASES)
    print("\nSUMMARY")
    print(f"Vector retrieval: {vector_hits}/{total} correct in top 3")
    print(f"Reranked retrieval: {reranked_hits}/{total} correct in top 3")
    print("Note: This is a small, manually labelled test set.")
if __name__ == "__main__":
    main()
