from app.test_new_db_answers import answer_with_test_db
TEST_CASES = [
    {
        "question": "I can't get into my ERP account.",
        "expected_status": "ANSWER GENERATED FROM CONTEXT",
    },
    {
        "question": "My fee payment went through but I cannot see the receipt.",
        "expected_status": "ANSWER GENERATED FROM DOCUMENTATION",
    },
    {
        "question": "How do I activate my library access?",
        "expected_status": "DOCUMENTED LIMITATION",
    },
    {
        "question": "Did my personal payment transaction succeed?",
        "expected_status": "LIVE RECORD NOT AVAILABLE",
    },
    {
        "question": "What is my examination result?",
        "expected_status": "LIVE RECORD NOT AVAILABLE",
    },
    {
        "question": "Who won the football match yesterday?",
        "expected_status": "NO RELEVANT CONTEXT",
    },
]
def main():
    passed = 0
    failed = 0
    print("\nADVANCED RAG EVALUATION")
    print("=" * 60)
    for i, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected = test["expected_status"]
        try:
            result = answer_with_test_db(question)
            if isinstance(result, tuple) and len(result) == 3:
                answer, sources, status = result
            else:
                print(f"\nTest {i}: Unexpected return format: {result!r}")
                failed += 1
                continue
            ok = status == expected
            print(f"\nTest {i}: {'PASS' if ok else 'CHECK'}")
            print(f"Question: {question}")
            print(f"Expected status: {expected}")
            print(f"Actual status: {status}")
            print(f"Sources: {sources}")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"\nTest {i}: ERROR")
            print(f"Question: {question}")
            print(f"Error: {exc}")
            failed += 1
    print("\n" + "=" * 60)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed/errors: {failed}")
    print("Review any failures before drawing conclusions.")
if __name__ == "__main__":
    main()
