
import streamlit as st
from app.test_new_db_answers import answer_with_test_db

st.set_page_config(
    page_title="ERP AI Support Chatbot — Test",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 ERP AI Support Chatbot")
st.caption("Test interface • Test database only")

st.info(
    "This portfolio chatbot answers using fictional support "
    "documentation. It cannot access live student records."
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display metadata for assistant messages
        if message["role"] == "assistant":
            if message.get("status"):
                st.caption(
                    f"Support check: {message['status']}"
                )

            if message.get("sources"):
                st.caption(
                    "Source documents: "
                    + ", ".join(message["sources"])
                )

# Chat input
question = st.chat_input("Ask an ERP support question...")

if question:
    # Display and save user question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Checking support documentation..."):
            try:
                result = answer_with_test_db(question)

                # Expected return:
                # (answer, sources, status)
                if isinstance(result, tuple) and len(result) == 3:
                    answer, sources, status = result

                    # Ensure consistent display types
                    answer = str(answer)
                    sources = list(sources) if sources else []
                    status = str(status)

                else:
                    # Fallback if the function returns a string
                    answer = str(result)
                    sources = []
                    status = "STATUS NOT PROVIDED"

                st.markdown(answer)

                st.caption(f"Support check: {status}")

                if sources:
                    st.caption(
                        "Source documents: "
                        + ", ".join(sources)
                    )
                else:
                    st.caption("Source documents: None")

                # Save response and metadata for chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "status": status,
                })

            except Exception:
                st.error(
                    "The test chatbot encountered an error. "
                    "Check the terminal for details."
                )
                st.exception(Exception(
                    "Review the terminal output for the original error."
                ))

# Clear conversation
if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
