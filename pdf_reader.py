from pypdf import PdfReader

from embeddings import EmbeddingManager
from rag_storage import RAGStorage


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# Load PDF
reader = PdfReader("data/hr_policy.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()

    if extracted:
        text += extracted


print(f"Total text length: {len(text)}")


# Chunk text
chunks = chunk_text(text)

print(f"Total chunks created: {len(chunks)}")


# Initialize systems
manager = EmbeddingManager()
storage = RAGStorage()


# Store chunks
for index, chunk in enumerate(chunks):

    print(f"Processing chunk {index + 1}/{len(chunks)}")

    embedding = manager.get_embedding(chunk)

    doc_id = storage.insert_document(
        content=chunk,
        embedding=embedding,
        metadata={
            "source": "hr_policy.pdf",
            "chunk_index": index
        }
    )

    print(f"Inserted document ID: {doc_id}")


print("PDF ingestion completed successfully!")