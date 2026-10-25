
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
    collection_name="erp_support"
)

question = "How can a student get a fee payment receipt?"
results = db.similarity_search(question, k=3)

print("\nQuestion:", question)
print("\nRetrieved support information:\n")

for doc in results:
    print("Source:", doc.metadata.get("source", "Unknown"))
    print(doc.page_content)
    print("-" * 50)
