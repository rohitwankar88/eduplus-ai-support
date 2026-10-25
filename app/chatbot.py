
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_name="erp_support"
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an ERP support assistant.

Use ONLY the support context below.
Include relevant troubleshooting steps in their original order.
Do not invent procedures or claim to have performed actions.

Never invent payment status, attendance changes, or exam results.
Never request passwords or OTPs.

If the context does not contain the answer, say:
"I don't have enough information to confirm that.
Please contact the authorized support team."

Support context:
{context}

Question: {question}

Answer in simple, clear English.
""")


def answer_question(question, return_result=False):
    question_lower = question.lower()

    # Safety check: personal records cannot be verified by this chatbot.
    personal_record_terms = [
         "my payment",
        "has my payment",
        "payment been successful",
        "is my payment successful",
        "my payment status",
        "my balance",
        "my refund",
        "what is my attendance",
        "show me my attendance",
        "tell me my attendance",
        "my current attendance",
        "my marks",
        "my exam result",
        "my result",
        "what marks did i get",
        "what score did i get",
        "my exam score"
    ]
    personal_record_terms = personal_record_terms + [
        "final exam marks",
        "marks in my final exam",
        "what marks did i get in my final exam"
    ]
    if any(term in question_lower for term in personal_record_terms):
        result = {
            "answer": (
                "I cannot access or verify your personal records. "
                "Please check the ERP portal or contact the "
                "authorized support team."
            ),
            "source": "Safety rule"
        }

        if return_result:
            return result

        print("\nChatbot:", result["answer"])
        print("\nSource:", result["source"])
        return

    # Retrieve relevant support documents.
    results = db.similarity_search_with_relevance_scores(
        question,
        k=3
    )

    docs = [
        doc for doc, score in results
        if score >= 0.35
    ]

    if not docs:
        result = {
            "answer": (
                "I couldn't find relevant support information. "
                "Please contact the authorized support team."
            ),
            "source": "None"
        }

        if return_result:
            return result

        print("\nChatbot:", result["answer"])
        print("\nSource: None")
        return

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    source = ", ".join(
        dict.fromkeys(
            Path(
                doc.metadata.get("source", "Unknown")
            ).name
            for doc in docs
        )
    )

    # Generate an answer using the retrieved context.
    response = llm.invoke(
        prompt.format_messages(
            context=context,
            question=question
        )
    )

    result = {
        "answer": response.content,
        "source": source
    }

    if return_result:
        return result

    print("\nChatbot:", result["answer"])
    print("\nSource:", result["source"])


if __name__ == "__main__":
    print("ERP Support Chatbot (type 'exit' to quit)")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if question:
            answer_question(question)
