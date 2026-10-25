
from app.test_new_db_answers import answer_with_test_db

# Evaluation questions for the test chatbot.
# These are fictional portfolio tests, not real student records.

test_cases = [
    {
        "question": "How do I troubleshoot login problems?",
        "expected_status": "ANSWER GENERATED FROM DOCUMENTATION",
        "expected_source": "login_support.txt",
        "expected_text": "password-reset",
    },
    {
        "question": "Where can a student find a fee payment receipt?",
        "expected_status": "ANSWER GENERATED FROM DOCUMENTATION",
        "expected_source": "fee_support.txt",
        "expected_text": "Payment History",
    },
    {
        "question": "How should staff troubleshoot a missing attendance record?",
        "expected_status": "ANSWER GENERATED FROM DOCUMENTATION",
        "expected_source": "attendance_support.txt",
        "expected_text": "synchronization",
    },
    {
        "question": "How long does library account activation take?",
        "expected_status": "ANSWER GENERATED FROM DOCUMENTATION",
        "expected_source": "library_support.txt",
        "expected_text": "two working days",
    },
    {
        "question": "How do I activate my library access?",
        "expected_status": "DOCUMENTED LIMITATION",
        "expected_source": "library_support.txt",
        "expected_text": "not documented",
    },
    {
        "question": "Did my transaction go through?",
        "expected_status": "LIVE RECORD NOT AVAILABLE",
        "expected_source": None,
        "expected_text": "cannot access live student records",
    },
    {
        "question": "What is my examination result?",
        "expected_status": "LIVE RECORD NOT AVAILABLE",
        "expected_source": None,
        "expected_text": "cannot access live student records",
    },
]

passed = 0
failed = 0

print("\nRAG CHATBOT EVALUATION")
print("=" * 60)

for number, case in enumerate(test_cases, start=1):
    question = case["question"]

    try:
        result = answer_with_test_db(question)

        if isinstance(result, tuple) and len(result) == 3:
            answer, sources, status = result
        else:
            answer = str(result)
            sources = []
            status = "UNKNOWN"

        answer = str(answer)
        sources = list(sources) if sources else []
        status = str(status)

        checks = {
            "status": status == case["expected_status"],
            "source": (
                case["expected_source"] in sources
                if case["expected_source"]
                else len(sources) == 0
            ),
            "answer": case["expected_text"].lower() in answer.lower(),
        }

        success = all(checks.values())

        if success:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {number}: {question}")
        print(f"Result: {'PASS' if success else 'FAIL'}")
        print(f"Status: {status}")
        print(f"Sources: {sources}")

        if not success:
            print("Failed checks:")
            for check_name, check_passed in checks.items():
                if not check_passed:
                    print(f" - {check_name}")

    except Exception as error:
        failed += 1
        print(f"\nTest {number}: {question}")
        print("Result: ERROR")
        print(f"Details: {error}")

total = passed + failed
accuracy = (passed / total * 100) if total else 0

print("\n" + "=" * 60)
print(f"Total tests: {total}")
print(f"Passed: {passed}")
print(f"Failed/errors: {failed}")
print(f"Evaluation pass rate: {accuracy:.1f}%")
print("=" * 60)
print("Original chatbot and database have not been modified.")
