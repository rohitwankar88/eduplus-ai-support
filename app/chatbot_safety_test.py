
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. CONFIGURATION — TEST DATABASE ONLY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db_test_step11"
COLLECTION_NAME = "erp_support_test"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
RELEVANCE_THRESHOLD = 0.35
TOP_K = 2


# ============================================================
# 2. INITIALIZE DATABASE, EMBEDDINGS AND LLM
# ============================================================

def initialize_database():
    """Connect to the isolated test Chroma database."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def initialize_llm():
    """Initialize the local language model."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


db = initialize_database()
llm = initialize_llm()


# ============================================================
# 3. PROMPT AND STANDARD RESPONSES
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are an ERP support assistant.

Use only the supplied support context.
Follow documented troubleshooting steps in their original order.
Do not invent procedures or claim to have performed actions.

Never invent payment status, attendance changes, or exam results.
Never request passwords or OTPs.
Never disclose another person's private records.

If the context does not contain the answer, say:
"I don't have enough information to confirm that.
Please contact the authorized support team."

Support context:
{context}

Question: {question}

Answer in simple, clear English.
""")

LIVE_RECORD_RESPONSE = (
    "I cannot access or verify live student records, including "
    "payment status, attendance records, or exam results. "
    "Please check the ERP portal or contact the authorized support team."
)

PRIVACY_RESPONSE = (
    "I cannot disclose another student's private records. "
    "Please contact the authorized support team for assistance."
)

NO_CONTEXT_RESPONSE = (
    "I couldn't find relevant support information. "
    "Please contact the authorized support team."
)

GENERATION_ERROR_RESPONSE = (
    "I couldn't generate an answer right now. "
    "Please contact the authorized support team."
)


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize(question):
    """Normalize case and common apostrophe variants."""
    return (
        question.lower()
        .replace("’", "'")
        .replace("‘", "'")
    )


# ============================================================
# 5. PRIVACY CHECK
# ============================================================

def asks_for_other_student_records(question):
    """Identify requests for another person's private ERP records."""
    q = normalize(question)

    other_person_terms = [
        "another student",
        "other student",
        "different student",
        "another student's",
        "other student's",
        "different student's",
        "other students",
        "someone else's",
        "someone elses",
        "another learner",
        "other learner",
        "different learner",
    ]

    record_terms = [
        "fee", "fees", "payment", "receipt", "transaction",
        "attendance", "exam result", "marks", "grade",
        "student record", "personal details", "balance",
    ]

    return (
        any(term in q for term in other_person_terms)
        and any(term in q for term in record_terms)
    )


# ============================================================
# 6. ATTENDANCE NAVIGATION CHECK
# ============================================================

def asks_attendance_viewing(question):
    """Identify general questions about how to view attendance."""
    q = normalize(question)

    how_to_phrases = [
        "how can i check",
        "how do i check",
        "how can i view",
        "how do i view",
        "how can i see",
        "how do i see",
        "how to check",
        "how to view",
        "how to see",
        "where can i check",
        "where do i check",
        "where can i view",
        "where do i view",
        "where can i see",
        "where do i see",
        "steps to check",
        "steps to view",
    ]

    return (
        "attendance" in q
        and any(phrase in q for phrase in how_to_phrases)
    )


def handle_attendance_question():
    """Provide general attendance navigation instructions."""
    return {
        "answer": (
            "To check your attendance:\n\n"
            "1. Log in to the ERP portal using your authorized student account.\n"
            "2. Open the Attendance section from the dashboard.\n"
            "3. Select the relevant semester or date range, if available.\n"
            "4. Review your attendance percentage and subject-wise history.\n"
            "5. If information is missing or incorrect, contact authorized support staff.\n\n"
            "I cannot access or verify live student records."
        ),
        "source": "General attendance guidance",
    }


# ============================================================
# 7. PERSONAL LIVE-RECORD CHECK
# ============================================================

def asks_for_live_record(question):
    """Identify requests to verify personal live ERP records."""
    q = normalize(question)

    payment_terms = [
        "payment", "fee", "fees", "paid", "transaction",
        "money deducted", "payment gateway",
    ]

    payment_status_terms = [
        "status", "confirmed", "confirmation", "verified",
        "verify", "received", "processed", "went through",
        "gone through", "go through", "successful", "succeeded",
        "completed", "cleared", "credited", "debited",
    ]

    personal_reference = ["my", "mine", "for me"]

    personal_payment_status = (
        any(term in q for term in payment_terms)
        and any(term in q for term in payment_status_terms)
        and any(term in q for term in personal_reference)
    )

    explicit_payment_status = any(
        phrase in q
        for phrase in [
            "has my payment been",
            "was my payment",
            "is my payment",
            "did my payment",
            "has my transaction",
            "did my transaction go through",
            "has my transaction gone through",
            "was the payment confirmed",
            "fee payment status",
            "payment status for me",
            "payment been verified",
            "payment been successful",
        ]
    )

    personal_exam_phrases = [
        "my exam result",
        "my examination result",
        "my results",
        "my marks",
        "my grades",
        "my score",
        "my exam score",
        "what marks did i get",
        "what score did i get",
        "show me my result",
        "tell me my result",
        "what is my result",
        "what are my results",
        "check my result",
        "check my marks",
    ]

    personal_exam_record = any(
        phrase in q for phrase in personal_exam_phrases
    )

    attendance_record_phrases = [
        "my attendance record",
        "my current attendance",
        "my attendance status",
        "my attendance percentage",
        "my attendance history",
        "my updated attendance",
        "my corrected attendance",
        "attendance record for me",
        "what is my attendance",
        "what's my attendance",
        "show me my attendance",
        "tell me my attendance",
        "give me my attendance",
        "my attendance today",
    ]

    personal_attendance_record = any(
        phrase in q for phrase in attendance_record_phrases
    )

    # General how-to attendance questions are allowed.
    if asks_attendance_viewing(question):
        personal_attendance_record = False

    return (
        personal_payment_status
        or explicit_payment_status
        or personal_exam_record
        or personal_attendance_record
    )


# ============================================================
# 8. LOGIN TROUBLESHOOTING
# ============================================================

def is_login_question(question):
    """Identify common ERP login troubleshooting questions."""
    q = normalize(question)

    login_terms = [
        "cannot log in",
        "can't log in",
        "cannot login",
        "can't login",
        "unable to log in",
        "unable to login",
        "login issue",
        "login problem",
        "trouble logging in",
        "trouble logging into",
        "login not working",
        "can't access my account",
    ]

    return any(term in q for term in login_terms)


def handle_login_question():
    """Return the documented login troubleshooting procedure."""
    return {
        "answer": (
            "To troubleshoot ERP portal login issues:\n\n"
            "1. Verify the username and registered account status.\n"
            "2. Check whether you are accessing the correct portal.\n"
            "3. Use the official password-reset process if needed.\n"
            "4. Check for account lockout or service outages.\n"
            "5. If unresolved, contact the authorized support team.\n\n"
            "Never share your password or OTP."
        ),
        "source": "login_support.txt",
    }


# ============================================================
# 9. DOCUMENT RETRIEVAL
# ============================================================

def retrieve_support_context(question):
    """Retrieve relevant documents and print diagnostic scores."""
    results = db.similarity_search_with_relevance_scores(
        question,
        k=TOP_K,
    )

    print("\n--- Retrieval diagnostics ---")

    for doc, score in results:
        source = Path(
            doc.metadata.get("source", "Unknown")
        ).name
        print(f"Source: {source} | Score: {score:.3f}")

    relevant_docs = [
        (doc, score)
        for doc, score in results
        if score >= RELEVANCE_THRESHOLD
    ]

    if not relevant_docs:
        return "", "None"

    context = "\n\n".join(
        doc.page_content for doc, score in relevant_docs
    )

    sources = list(dict.fromkeys(
        Path(doc.metadata.get("source", "Unknown")).name
        for doc, score in relevant_docs
    ))

    return context, ", ".join(sources)


# ============================================================
# 10. LLM ANSWER GENERATION
# ============================================================

def generate_answer(question, context, source):
    """Generate an answer grounded in the retrieved context."""
    try:
        response = llm.invoke(
            prompt.format_messages(
                context=context,
                question=question,
            )
        )

        answer = response.content.strip()

        if not answer:
            answer = (
                "I don't have enough information to confirm that. "
                "Please contact the authorized support team."
            )

        return {
            "answer": answer,
            "source": source,
        }

    except Exception:
        return {
            "answer": GENERATION_ERROR_RESPONSE,
            "source": source,
        }


# ============================================================
# 11. MAIN QUESTION ROUTER
# ============================================================

def answer_question(question, return_result=False):
    """
    Route a question through safety checks, deterministic handlers,
    retrieval, and answer generation.
    """
    q = normalize(question)

    # Priority 1: Protect other people's private records.
    if asks_for_other_student_records(question):
        result = {
            "answer": PRIVACY_RESPONSE,
            "source": "Safety rule",
        }

    # Priority 2: Allow general attendance navigation.
    elif asks_attendance_viewing(question):
        result = handle_attendance_question()

    # Priority 3: Block personal live-record verification.
    elif asks_for_live_record(question):
        result = {
            "answer": LIVE_RECORD_RESPONSE,
            "source": "Safety rule",
        }

    # Priority 4: Use documented login troubleshooting.
    elif is_login_question(question):
        result = handle_login_question()

    # Priority 5: Answer documented library activation timing.
    elif (
        "library" in q
        and "activat" in q
        and any(term in q for term in [
            "how long", "how many days", "duration", "takes",
            "time required", "waiting period",
        ])
    ):
        result = {
            "answer": (
                "Library account activation takes two working days."
            ),
            "source": "library_support.txt",
        }

    # Priority 6: Explain that the activation procedure is undocumented.
    elif "library" in q and "activat" in q:
        result = {
            "answer": (
                "The library account activation procedure is not documented "
                "in the available support information. Please contact the "
                "authorized support team for assistance."
            ),
            "source": "library_support.txt",
        }

    # Priority 7: Retrieve other support information.
    else:
        try:
            context, source = retrieve_support_context(question)

            if not context:
                result = {
                    "answer": NO_CONTEXT_RESPONSE,
                    "source": "None",
                }
            else:
                result = generate_answer(
                    question,
                    context,
                    source,
                )

        except Exception:
            result = {
                "answer": GENERATION_ERROR_RESPONSE,
                "source": "Test database",
            }

    # Return the result when called by a test or another function.
    if return_result:
        return result

    # Display the result in the terminal chatbot.
    print("\nChatbot:", result["answer"])
    print("\nSource:", result["source"])

    return result


# ============================================================
# 12. TERMINAL CHAT LOOP
# ============================================================

def main():
    """Run the test chatbot in the terminal."""
    print("ERP Support Chatbot TEST COPY (type 'exit' to quit)")

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Test chatbot closed.")
                break

            if question:
                answer_question(question)

        except (KeyboardInterrupt, EOFError):
            print("\nTest chatbot closed.")
            break


if __name__ == "__main__":
    main()
