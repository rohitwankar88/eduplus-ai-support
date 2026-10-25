
from app.chatbot import answer_question

TEST_CASES = [
    {
        "question": "How long does library account activation take?",
        "expected_source": "library_support.txt",
        "expected_answer": "two working days",
    },
    {
        "question": "How can a student troubleshoot a login problem?",
        "expected_source": "login_support.txt",
        "expected_answer": "authorized support team",
    },
    {
        "question": "How can a student correct an attendance record?",
        "expected_source": "attendance_support.txt",
        "expected_answer": None,
    },
    {
        "question": "What is the student's current fee payment status?",
        "expected_source": None,
        "expected_answer": "authorized support team",
        "must_not_contain": ["your payment was successful"],
    },
    {
        "question": "What was the student's final exam score?",
        "expected_source": "exam_support.txt",
        "expected_answer": "final exam score",
        "must_not_contain": ["your score was"],
    },
    {
        "question": "Where can a student get a fee payment receipt?",
        "expected_source": "fee_support.txt",
        "expected_answer": "payment history",
        "must_not_contain": ["your payment was successful"],
    },
]


def run_evaluation():
    passed = 0

    print("\nERP Support RAG Evaluation")
    print("=" * 50)

    for number, test in enumerate(TEST_CASES, start=1):
        result = answer_question(
            test["question"],
            return_result=True
        )

        answer = result["answer"].lower()
        source = result["source"].lower()
        checks = []

        if test["expected_source"]:
            checks.append(
                test["expected_source"].lower() in source
            )

        if test["expected_answer"]:
            checks.append(
                test["expected_answer"].lower() in answer
            )

        if test.get("must_not_contain"):
            checks.append(
                not any(
                    phrase.lower() in answer
                    for phrase in test["must_not_contain"]
                )
            )

        success = all(checks)

        if success:
            passed += 1

        print(f"\nTest {number}: {'PASS' if success else 'FAIL'}")
        print(f"Question: {test['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Source: {result['source']}")

    total = len(TEST_CASES)
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    print(f"Pass rate: {passed / total * 100:.1f}%")


if __name__ == "__main__":
    run_evaluation()
