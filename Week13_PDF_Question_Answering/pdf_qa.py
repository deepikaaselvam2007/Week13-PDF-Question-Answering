from pathlib import Path
import csv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "dataset" / "sample.pdf"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "week13_pdf_knowledge"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

def main():
    # 1. Extract text from the PDF.
    reader = PdfReader(str(PDF_PATH))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages)
    (ROOT / "dataset" / "extracted_text.txt").write_text(full_text, encoding="utf-8")
    print(f"Extracted text from {len(pages)} PDF pages.")

    # 2. Split the text into overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.create_documents(pages, metadatas=[
        {"source": PDF_PATH.name, "page": page_no + 1} for page_no in range(len(pages))
    ])
    with (ROOT / "dataset" / "chunks.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "page", "chunk_text"])
        for i, chunk in enumerate(chunks, 1):
            writer.writerow([i, chunk.metadata.get("page", ""), chunk.page_content])

    # 3. Embed and store chunks in ChromaDB.
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION, persist_directory=str(DB_DIR),
            embedding_function=embeddings
        )
        vectorstore.delete_collection()
    except Exception:
        pass
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=COLLECTION,
        persist_directory=str(DB_DIR)
    )

    # 4. Connect to the local Ollama model.
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)

    # 5. Retrieve Top-5 chunks and ask the LLM to answer from context only.
    with (ROOT / "dataset" / "queries.csv").open(encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))
    answer_rows = []
    for item in questions:
        question = item["question"]
        matches = vectorstore.similarity_search_with_score(question, k=5)
        context = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d, score in matches
        )
        prompt = f"""You are a document question-answering assistant.
Answer the question using ONLY the context below. If the answer is not present,
say: 'The provided document does not contain enough information.'
Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        answer = llm.invoke(prompt)
        retrieved = " || ".join(
            f"Page {d.metadata.get('page', '?')}: {d.page_content[:180].replace(chr(10), ' ')}"
            for d, score in matches
        )
        answer_rows.append({
            "question_id": item["question_id"], "question": question,
            "answer": str(answer).strip(), "top5_retrieved_chunks": retrieved
        })
        print(f"\nQ: {question}\nA: {str(answer).strip()}\n")

    out = ROOT / "dataset" / "answers.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(answer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(answer_rows)
    print(f"Saved {len(answer_rows)} answers to {out}")

if __name__ == "__main__":
    main()
