# ERP Support Assistant — RAG Chatbot
A local AI-powered ERP support assistant that retrieves information from a knowledge base and generates answers using a language model. The project uses Retrieval-Augmented Generation (RAG) to ground responses in support documentation.
## Features
- Ask questions about ERP support procedures.
- Retrieve relevant information from a ChromaDB vector database.
- Upload PDF and TXT support documents through a Streamlit interface.
- Extract PDF text with `pypdf`.
- Split documents into chunks of 500 characters with an 80-character overlap.
- Store source filename and page metadata for retrieved documents.
- Detect duplicate uploads using SHA-256 file hashes.
- Use a local Ollama model for answer generation and embeddings.
- Apply prompt rules to reduce unsupported claims about payments, attendance, and exam results.
## Technology Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- Ollama
- `llama3.2:3b` — language model
- `nomic-embed-text` — embedding model
- `pypdf` — PDF text extraction
## Architecture
1. A user submits a question through the Streamlit interface.
2. The application searches the ChromaDB knowledge base for relevant support context.
3. The retrieved context is passed to the local language model.
4. The model generates an answer based on the supplied context and the prompt's safety instructions.
5. Users can upload PDF or TXT files, which are extracted, chunked, embedded, and indexed in the original support collection.
## Project Structure
```text
eduplus-ai-support/
├── app/
│   ├── chatbot.py
│   ├── ui.py
│   ├── ingest_upload.py
│   ├── document_processor.py
│   ├── test_new_db_answers.py
│   ├── test_ui.py
│   └── evaluate_expanded_test.py
├── data/
│   ├── attendance_support.txt
│   ├── exam_support.txt
│   ├── fee_support.txt
│   ├── library_support.txt
│   └── login_support.txt
├── chroma_db/                  # Original knowledge base (generated locally)
├── chroma_db_test_step11/      # Test database (generated locally)
├── uploaded_docs/              # Temporary PDF files
├── .env                        # Local configuration/secrets, if needed
└── README.md

## Setup and Run

### Prerequisites
- Python 3.10 or newer
- Ollama installed and running
- Models: llama3.2:3b and nomic-embed-text

### Install dependencies
```powershell
pip install -r requirements.txt
```

### Download models
```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Start the application
```powershell
python -m streamlit run app\ui.py
```

Ensure your local ChromaDB knowledge base has been created and populated before running the application.
