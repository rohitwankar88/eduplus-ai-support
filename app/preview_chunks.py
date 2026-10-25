
from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def split_fee_document(text):
    header, body = text.split("Issue 1:", 1)

    issue1, issue2_and_rest = body.split("Issue 2:", 1)

    issue1_section = (
        header.strip()
        + "\n\nIssue 1: "
        + issue1.strip()
    )

    remaining = "Issue 2:" + issue2_and_rest

    other_sections = re.split(
        r"(?=Safety rules:|Note:)",
        remaining
    )

    sections = [issue1_section]
    sections.extend(
        section.strip()
        for section in other_sections
        if section.strip()
    )

    return sections


def main():
    total_chunks = 0

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if file_path.name == "fee_support.txt":
            sections = split_fee_document(text)
            chunks = []

            for section in sections:
                chunks.extend(splitter.split_text(section))
        else:
            chunks = splitter.split_text(text)

        total_chunks += len(chunks)

        print(f"\n--- {file_path.name}: {len(chunks)} chunks ---")

        for index, chunk in enumerate(chunks, start=1):
            preview = " ".join(chunk.split())[:180]
            print(f"\nChunk {index} ({len(chunk)} characters)")
            print(preview)

    print(f"\nTotal proposed chunks: {total_chunks}")
    print("\nPreview only. No files or database were modified.")


if __name__ == "__main__":
    main()
