
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

THRESHOLD = 0.45
TOP_K = 5
MAX_SCORE_GAP = 0.25

LIVE_RECORD_RESPONSE = (
    "I cannot access live student records to verify that "
    "information. Please contact the authorized support team."
)

NO_RELEVANT_CONTEXT = (
    "I couldn't find relevant support information. "
    "Please contact the authorized support team."
)

FALLBACK = (
    "I don't have enough information to confirm that. "
    "Please contact the authorized support team."
)

MISSING_RECEIPT_RESPONSE = (
    "The available support documentation does not provide "
    "enough relevant information to answer this receipt "
    "question. Please contact the authorized support team."
)


# ============================================================
# 2. INITIALIZE EMBEDDINGS, DATABASE AND LLM
# ============================================================

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# ============================================================
# 3. QUESTION CLASSIFICATION HELPERS
# ============================================================

def asks_about_missing_receipt(question):
    """Identify questions about missing or unavailable receipts."""

    q = question.lower().replace("’", "'")

    receipt_terms = [
        "receipt",
        "invoice",
        "payment proof",
        "payment confirmation slip",
    ]

    missing_terms = [
        "missing",
        "not generated",
        "not received",
        "cannot download",
        "can't download",
        "unable to download",
        "cannot find",
        "can't find",
        "unavailable",
        "failed",
        "didn't get",
        "did not get",
        "do not have",
        "don't have",
        "haven't received",
        "have not received",
        "not showing",
        "doesn't show",
        "does not show",
        "not appearing",
        "can't access",
        "cannot access",
        "unable to access",
        "not available",
    ]

    return (
        any(term in q for term in receipt_terms)
        and any(term in q for term in missing_terms)
    )


def asks_for_live_record(question):
    """
    Detect requests for personal live ERP records.
    General troubleshooting questions should remain answerable.
    """

    q = question.lower().replace("’", "'")

    payment_terms = [
        "payment",
        "fee",
        "fees",
        "paid",
        "transaction",
        "money deducted",
        "payment gateway",
    ]

    payment_status_terms = [
        "status",
        "confirmed",
        "confirmation",
        "verified",
        "verify",
        "received",
        "processed",
        "went through",
        "go through",
        "successful",
        "succeeded",
        "completed",
        "cleared",
        "credited",
        "debited",
    ]

    personal_reference = [
        "my",
        "mine",
        "student's",
        "students'",
        "for me",
    ]

    asks_payment_status = (
        any(term in q for term in payment_terms)
        and any(term in q for term in payment_status_terms)
        and any(term in q for term in personal_reference)
    )

    exam_terms = [
        "exam",
        "examination",
        "result",
        "marks",
        "grade",
        "score",
        "passed",
        "failed",
    ]

    asks_exam_record = (
        any(term in q for term in exam_terms)
        and any(term in q for term in personal_reference)
    )

    attendance_terms = [
        "attendance",
        "present",
        "absent",
    ]

    attendance_record_terms = [
        "current",
        "updated",
        "corrected",
        "marked",
        "record",
        "status",
    ]

    asks_attendance_record = (
        any(term in q for term in attendance_terms)
        and any(term in q for term in attendance_record_terms)
        and any(term in q for term in personal_reference)
    )

    # Missing receipt troubleshooting must not be mistaken
    # for a request to verify payment success.
    if asks_about_missing_receipt(question):
        asks_payment_status = False

    return (
        asks_payment_status
        or asks_exam_record
        or asks_attendance_record
    )


def asks_library_activation_time(question):
    """Identify library activation timeframe questions."""

    q = question.lower()

    asks_time = any(
        term in q
        for term in [
            "how long",
            "how many days",
            "how many working days",
            "how much time",
            "timeframe",
            "duration",
            "take to activate",
            "activation time",
            "working days",
        ]
    )

    return "library" in q and asks_time


def asks_library_activation_procedure(question):
    """Identify library activation procedure questions."""

    q = question.lower()

    asks_procedure = any(
        term in q
        for term in [
            "how do i activate",
            "how to activate",
            "activate my library",
            "activation procedure",
            "activation process",
            "enable my library",
            "steps to activate",
        ]
    )

    return "library" in q and asks_procedure


def asks_login_troubleshooting(question):
    """Identify login troubleshooting questions."""

    q = question.lower().replace("’", "'")

    login_terms = [
        "login",
        "log in",
        "sign in",
        "signin",
    ]

    issue_terms = [
        "troubleshoot",
        "problem",
        "issue",
        "cannot",
        "can't",
        "unable",
        "not working",
        "failed",
        "failure",
        "locked",
        "forgot password",
        "password reset",
    ]

    return (
        any(term in q for term in login_terms)
        and any(term in q for term in issue_terms)
    )


def asks_attendance_troubleshooting(question):
    """Identify general attendance issue/correction questions."""

    q = question.lower()

    attendance_terms = [
        "attendance",
        "attendance record",
    ]

    issue_terms = [
        "error",
        "incorrect",
        "missing",
        "wrong",
        "correct",
        "fix",
        "troubleshoot",
        "problem",
        "issue",
        "not showing",
        "update",
        "synchronization",
        "synchronized",
        "synchronised",
        "synchronize",
        "synchronise",
        "sync",
    ]

    return (
        any(term in q for term in attendance_terms)
        and any(term in q for term in issue_terms)
    )


# ============================================================
# 4. RETRIEVE RELEVANT DOCUMENTATION
# ============================================================


def retrieve_context(question):
    """
    Retrieve relevant chunks using source metadata filtering
    and relevance-score filtering.
    """

    question_lower = question.lower()

    # Identify the relevant document category from the question.
    source_filters = {
        "login_support.txt": [
            "login", "log in", "sign in", "signin",
            "password", "locked out",
        ],
        "fee_support.txt": [
            "fee", "payment", "receipt", "transaction",
            "paid", "pay",
        ],
        "attendance_support.txt": [
            "attendance", "absent", "synchroniz",
        ],
        "library_support.txt": [
            "library", "activation",
        ],
        "exam_support.txt": [
            "exam", "examination", "marks", "result",
        ],
    }

    matching_sources = [
        source
        for source, keywords in source_filters.items()
        if any(keyword in question_lower for keyword in keywords)
    ]

    # Filter by source only when a category is identified.
    search_kwargs = {"k": TOP_K}

    if len(matching_sources) == 1:
        search_kwargs["filter"] = {
            "source": matching_sources[0]
        }

    results = db.similarity_search_with_relevance_scores(
        question,
        **search_kwargs,
    )

    if not results:
        return "", []

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

    if not qualifying:
        return "", []

    context_parts = []
    sources = []

    for doc, score in qualifying:
        source = Path(
            str(doc.metadata.get("source", "unknown"))
        ).name

        context_parts.append(
            f"[Source: {source}]\n{doc.page_content}"
        )

        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


# 5. ANSWER PROMPT
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a careful support assistant for a fictional
educational institution's ERP system.

Answer only from the supplied retrieved documentation.

STRICT RULES:

1. Never invent URLs, menu items, buttons, procedures,
   IDs, PINs, passwords, or system capabilities.

2. Never claim to access live ERP records.

3. Never claim that a student's payment, examination
   result, or attendance record has been verified.

4. Refer individual live-record verification to
   the authorized support team.

5. Do not invent undocumented library activation procedures.

6. If documentation says library activation takes two
   working days, give that exact timeframe.

7. For finding fee receipts, use documented navigation only.

8. Missing-receipt troubleshooting may include verifying
   authorized payment records, checking ERP payment updates,
   checking receipt-generation status, checking logs only
   when authorized, and creating a support ticket.

9. Troubleshooting does not prove that payment succeeded.

10. Never request passwords or OTPs.

11. Attendance corrections must be escalated to authorized
    staff. Never claim to modify attendance.

12. If the context is insufficient, say so and refer the
    user to authorized support.

13. Use clear, concise English.

14. The documents are fictional portfolio data, not
    official institutional policy.
            """,
        ),
        (
            "human",
            """
Question:
{question}

Retrieved documentation:
{context}

Answer using only the available documentation.
            """,
        ),
    ]
)

answer_chain = answer_prompt | llm


# ============================================================
# 6. DETERMINISTIC SUPPORT ANSWERS
# ============================================================

def login_troubleshooting_answer(context, sources):
    """Return documented login troubleshooting steps."""

    context_lower = context.lower()

    has_login_source = any(
        source.lower() == "login_support.txt"
        for source in sources
    )

    required_phrases = [
        "verify the username",
        "correct portal",
        "password-reset process",
        "account lockout",
        "authorized support team",
    ]

    if (
        not has_login_source
        or not all(
            phrase in context_lower
            for phrase in required_phrases
        )
    ):
        return (
            NO_RELEVANT_CONTEXT,
            sources,
            "DOCUMENTATION INSUFFICIENT",
        )

    answer = (
        "According to the login troubleshooting documentation:\n\n"
        "1. Verify the username and registered account status.\n"
        "2. Check whether the user is accessing the correct portal.\n"
        "3. Use the official password-reset process if needed.\n"
        "4. Check for account lockout or service outages.\n"
        "5. Escalate unresolved issues to the authorized support team.\n\n"
        "Never share passwords or OTPs, and never bypass authentication."
    )

    return (
        answer,
        sources,
        "ANSWER GENERATED FROM DOCUMENTATION",
    )


def attendance_troubleshooting_answer(context, sources):
    """Return documented attendance troubleshooting steps."""

    context_lower = context.lower()

    has_attendance_source = any(
        source.lower() == "attendance_support.txt"
        for source in sources
    )

    required_phrases = [
        "student id and relevant date",
        "submitted by the authorized staff member",
        "pending synchronization",
        "without requesting unnecessary personal data",
        "escalate corrections to authorized staff",
    ]

    if (
        not has_attendance_source
        or not all(
            phrase in context_lower
            for phrase in required_phrases
        )
    ):
        return (
            NO_RELEVANT_CONTEXT,
            sources,
            "DOCUMENTATION INSUFFICIENT",
        )

    answer = (
        "According to the attendance troubleshooting documentation:\n\n"
        "1. Confirm the student ID and relevant date.\n"
        "2. Check whether attendance was submitted by the "
        "authorized staff member.\n"
        "3. Verify whether the record is pending synchronization.\n"
        "4. Collect relevant details without requesting "
        "unnecessary personal data.\n"
        "5. Escalate corrections to authorized staff.\n\n"
        "Attendance must not be modified without authorization."
    )

    return (
        answer,
        sources,
        "ANSWER GENERATED FROM DOCUMENTATION",
    )


def missing_receipt_answer(context, sources):
    """Return documented missing-receipt troubleshooting."""

    context_lower = context.lower()

    has_fee_source = any(
        source.lower() == "fee_support.txt"
        for source in sources
    )

    has_troubleshooting = (
        "verify the transaction status" in context_lower
        and "receipt generation is pending or failed"
        in context_lower
    )

    if not has_fee_source or not has_troubleshooting:
        return (
            MISSING_RECEIPT_RESPONSE,
            sources,
            "DOCUMENTATION INSUFFICIENT",
        )

    answer = (
        "The support documentation recommends these "
        "troubleshooting steps for a missing receipt:\n\n"
        "1. Ask authorized staff to verify the transaction "
        "status in authorized payment records.\n"
        "2. Check whether the ERP system received the payment update.\n"
        "3. Check whether receipt generation is pending or failed.\n"
        "4. Check application logs only if authorized.\n"
        "5. If unresolved, create a support ticket for investigation.\n\n"
        "These are general troubleshooting steps and do not "
        "confirm that your payment succeeded. Contact the "
        "authorized support team for assistance."
    )

    return (
        answer,
        sources,
        "ANSWER GENERATED FROM DOCUMENTATION",
    )


def fee_receipt_location_answer(context, sources):
    """Return receipt navigation only when the source supports it."""

    context_lower = context.lower()

    has_fee_source = any(
        source.lower() == "fee_support.txt"
        for source in sources
    )

    required_phrases = [
        "log in to the erp portal",
        "payment history or fee receipts",
        "select the relevant payment",
    ]

    if (
        not has_fee_source
        or not all(
            phrase in context_lower
            for phrase in required_phrases
        )
    ):
        return (
            NO_RELEVANT_CONTEXT,
            sources,
            "DOCUMENTATION INSUFFICIENT",
        )

    answer = (
        "According to the available support documentation:\n\n"
        "1. Log in to the ERP portal using your authorized account.\n"
        "2. Open the Fees or Fee Management section.\n"
        "3. Look for Payment History or Fee Receipts.\n"
        "4. Select the relevant payment to view the receipt.\n\n"
        "If the receipt is unavailable, contact the authorized "
        "support team. These general instructions do not verify "
        "an individual payment."
    )

    return (
        answer,
        sources,
        "ANSWER GENERATED FROM DOCUMENTATION",
    )


# ============================================================
# 7. MAIN ANSWER FUNCTION
# ============================================================

def answer_with_test_db(question):

    # A. Block requests for live individual records.
    if asks_for_live_record(question):
        return (
            LIVE_RECORD_RESPONSE,
            [],
            "LIVE RECORD NOT AVAILABLE",
        )

    # B. Retrieve relevant documentation.
    context, sources = retrieve_context(question)

    # C. Handle missing receipt questions.
    if asks_about_missing_receipt(question):
        return missing_receipt_answer(context, sources)

    # D. No relevant documentation.
    if not context.strip():
        return (
            NO_RELEVANT_CONTEXT,
            [],
            "NO RELEVANT CONTEXT",
        )

    q = question.lower()
    context_lower = context.lower()

    # E. Deterministic login troubleshooting.
    if asks_login_troubleshooting(question):
        return login_troubleshooting_answer(context, sources)

    # F. Deterministic attendance troubleshooting.
    if asks_attendance_troubleshooting(question):
        return attendance_troubleshooting_answer(context, sources)

    # G. Deterministic library activation timeframe.
    if asks_library_activation_time(question):
        has_library_source = any(
            source.lower() == "library_support.txt"
            for source in sources
        )

        if (
            has_library_source
            and "two working days" in context_lower
        ):
            return (
                "Library account activation takes two working "
                "days, according to the available support "
                "documentation.",
                sources,
                "ANSWER GENERATED FROM DOCUMENTATION",
            )

    # H. Library activation procedure is undocumented.
    if asks_library_activation_procedure(question):
        has_library_source = any(
            source.lower() == "library_support.txt"
            for source in sources
        )

        procedure_is_undocumented = (
            "does not describe the library account activation"
            in context_lower
            or "procedure is not documented" in context_lower
            or "procedure not documented" in context_lower
        )

        if has_library_source and procedure_is_undocumented:
            return (
                "The library account activation procedure is "
                "not documented in the available support "
                "information. Please contact the authorized "
                "support team for instructions.",
                sources,
                "DOCUMENTED LIMITATION",
            )

    # I. Finding a fee receipt.
    asks_receipt_location = (
        "receipt" in q
        and any(
            term in q
            for term in [
                "where",
                "find",
                "locate",
                "access",
                "get",
                "view",
            ]
        )
        and not asks_about_missing_receipt(question)
    )

    if asks_receipt_location:
        return fee_receipt_location_answer(context, sources)

    # J. Generate a grounded answer for other questions.
    try:
        response = answer_chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        answer = response.content.strip()

        if not answer:
            return (
                FALLBACK,
                sources,
                "EMPTY MODEL RESPONSE",
            )

        return (
            answer,
            sources,
            "ANSWER GENERATED FROM CONTEXT",
        )

    except Exception:
        # Avoid exposing internal exception details to users.
        return (
            FALLBACK,
            sources,
            "MODEL ERROR",
        )


# ============================================================
# 8. TEST SUITE
# ============================================================

if __name__ == "__main__":

    print("\nTESTING CHATBOT WITH NEW DATABASE")
    print(f"Test database: {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {db._collection.count()}")
    print(f"Relevance threshold: {THRESHOLD}")
    print(f"Maximum score gap: {MAX_SCORE_GAP}")

    questions = [
        "What is the student's current fee payment status?",
        "Has my fee payment been verified?",
        "What is my examination result?",
        "Tell me my exam marks.",
        "I paid my fees. Was the payment confirmed?",
        "Did my transaction go through?",
        "How do I activate my library access?",
        "How long does library account activation take?",
        "How many working days does library activation take?",
        "Where can a student find a fee payment receipt?",
        "What should be done if payment is successful but the receipt is not generated?",
        "I paid but cannot download my receipt.",
        "I haven't received my receipt.",
        "How do I troubleshoot login problems?",
        "I cannot log in to the ERP portal.",
        "I forgot my password and cannot sign in.",
        "How can an attendance error be corrected?",
        "What should I do if my attendance record is missing?",
        "How should staff troubleshoot a missing attendance record?",
        "What if attendance has not synchronized?",
    ]

    for question in questions:
        answer, sources, status = answer_with_test_db(question)

        print("\n" + "=" * 65)
        print(f"Question: {question}")
        print(f"Support check: {status}")
        print("\nAnswer:")
        print(answer)
        print(
            "\nSources:",
            ", ".join(sources) if sources else "None",
        )

    print("\n" + "=" * 65)
    print("Test completed.")
    print("Original chatbot and database remain unchanged.")
