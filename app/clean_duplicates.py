from collections import defaultdict
from app.ingest_upload import get_database

db = get_database()
data = db.get(include=["metadatas", "documents"])

seen = set()
duplicate_ids = []

for chunk_id, metadata, document in zip(
    data["ids"],
    data["metadatas"],
    data["documents"]
):
    source = metadata.get("source", "Unknown")

    if source != "erp_student_support_test.txt":
        continue

    key = (source, document.strip())

    if key in seen:
        duplicate_ids.append(chunk_id)
    else:
        seen.add(key)

print("Duplicate chunks found:", len(duplicate_ids))

if duplicate_ids:
    print("First duplicate IDs:", duplicate_ids[:5])
    confirm = input(
        "Type DELETE to remove only these duplicates: "
    )

    if confirm == "DELETE":
        db.delete(ids=duplicate_ids)
        print("Duplicate chunks removed.")
    else:
        print("No changes made.")
else:
    print("No duplicate chunks found.")