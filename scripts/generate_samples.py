"""Create the sample PDF and DOCX files in sample_docs/ (run once: python scripts/generate_samples.py)."""
from pathlib import Path

import pymupdf
from docx import Document

OUT = Path(__file__).resolve().parent.parent / "sample_docs"

RAG_PAGES = [
    "Retrieval-Augmented Generation (RAG)\n\n"
    "Large language models can write fluent text, but they only know what was in their training data. "
    "Retrieval-augmented generation fixes this by searching a document collection first and giving the "
    "retrieved passages to the model together with the question. The model then answers from those passages. "
    "This reduces made-up answers and lets the system cite its sources.\n\n"
    "A RAG system has two stages. In the indexing stage, documents are split into chunks, each chunk is turned "
    "into an embedding vector, and the vectors are stored in a vector index. In the query stage, the question "
    "is embedded, the nearest chunks are retrieved, and they are placed into the prompt.",
    "Improving retrieval quality\n\n"
    "Hybrid search combines keyword matching with vector similarity. Keyword search is precise for names and "
    "rare terms, while vector search finds passages that express the same idea in different words. Reciprocal "
    "rank fusion merges both rankings without needing to compare their scores directly.\n\n"
    "Agentic RAG lets an autonomous agent decide when to search, which query to use, and whether the evidence "
    "is good enough to answer. Such an agent may rewrite the query or search again after reading the first results.",
]
VDB = [
    ("Vector Databases", None),
    ("A vector database stores embeddings and finds the nearest neighbours of a query vector. Exact search compares "
     "the query with every stored vector. Approximate methods such as HNSW graphs or inverted file indexes trade a "
     "small loss in accuracy for a large gain in speed.", None),
    ("FAISS is an open source library from Meta for similarity search over dense vectors. A flat index is exact and "
     "works well for personal collections of a few hundred thousand passages. Larger collections benefit from "
     "compressed or approximate indexes.", None),
    ("Cosine similarity measures the angle between two vectors. When all vectors have unit length, cosine similarity "
     "equals the inner product, so an inner product index returns the most similar passages directly.", None),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pdf = pymupdf.open()
    for text in RAG_PAGES:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=11)
    pdf.save(OUT / "retrieval_augmented_generation.pdf")
    doc = Document()
    doc.add_heading(VDB[0][0], level=1)
    for text, _ in VDB[1:]:
        doc.add_paragraph(text)
    doc.save(OUT / "vector_databases.docx")
    print("Sample files written to", OUT)
