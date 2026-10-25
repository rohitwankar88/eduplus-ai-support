
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploaded_docs"

UPLOAD_DIR.mkdir(exist_ok=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def process_uploaded_file(uploaded_file):
    filename = Path(uploaded_file.name).name
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    content = uploaded_file.getvalue()

    if suffix == ".txt":
        text = content.decode("utf-8-sig")
        pages = [text]
    else:
        temporary_path = UPLOAD_DIR / filename
        temporary_path.write_bytes(content)

        reader = PdfReader(str(temporary_path))
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]

    documents = []

    for page_number, text in enumerate(pages, start=1):
        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_number
                }
            )
        )

    if not documents:
        raise ValueError(
            "No readable text found. "
            "Scanned PDFs may require OCR."
        )

    chunks = splitter.split_documents(documents)

    return chunks
