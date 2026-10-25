
import streamlit as st

from app.chatbot import answer_question
from app.ingest_upload import (
    index_uploaded_file,
    get_indexed_documents
)

st.set_page_config(
    page_title="ERP Support Assistant",
    page_icon="🎓"
)

st.title("🎓 ERP Support Assistant")
st.caption("AI-powered support using your local knowledge base")

# Document upload
st.sidebar.header("Upload Support Documents")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    if st.sidebar.button("Process document"):
        try:
            with st.spinner("Reading and indexing document..."):
                chunk_count = index_uploaded_file(uploaded_file)

            st.sidebar.success(
                f"Document indexed successfully! Chunks: {chunk_count}"
            )

        except ValueError as e:
            if "already been indexed" in str(e):
                st.sidebar.warning(
                    "This document has already been added. "
                    "No duplicate chunks were created."
                )
            else:
                st.sidebar.error(f"Upload failed: {e}")

        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")

# Display indexed documents
st.sidebar.divider()
st.sidebar.header("📚 Indexed Documents")

try:
    documents = get_indexed_documents()

    if documents:
        st.sidebar.caption(
            f"Total documents: {len(documents)}"
        )

        for document in documents:
            st.sidebar.write(f"📄 {document}")
    else:
        st.sidebar.info("No documents indexed yet.")

except Exception as e:
    st.sidebar.warning(
        f"Could not load documents: {e}"
    )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input(
    "Ask about fees, login, attendance, exams, or uploaded documents"
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching support documents..."):
            try:
                result = answer_question(
                    question,
                    return_result=True
                )

                st.markdown(result["answer"])
                st.caption("Source: " + result["source"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"]
                    + "\n\nSource: " + result["source"]
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
