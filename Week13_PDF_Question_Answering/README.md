# Week 13 – PDF Question Answering with LangChain, ChromaDB, and Ollama

## Aim
Extract PDF text, split it into chunks, create embeddings, retrieve the Top-5 relevant chunks, and generate context-aware answers using the local Ollama model.

## Included sample
`dataset/sample.pdf` is a 12-section AI/ML study guide generated for this assignment. You can run the project without searching for another PDF. If your instructor specifically requires an external 10–20 page PDF, replace `dataset/sample.pdf` with your selected PDF and keep the filename `sample.pdf`.

## Prerequisites
- Python 3.12 recommended.
- Ollama installed.
- A local Ollama model. This project uses `llama3.2`.

Check Ollama in a terminal:
- `ollama --version`
- `ollama list`
- If `llama3.2` is missing, run `ollama pull llama3.2`.

## Run in VS Code
1. Open this folder in VS Code.
2. Create a virtual environment: `py -3.12 -m venv .venv`
3. Activate: `.venv\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Make sure Ollama is running and `llama3.2` is available.
6. Run: `python pdf_qa.py`

The first run may download the embedding model. Ollama answers are generated locally and may take some time depending on your computer.

## Output
- `dataset/extracted_text.txt`
- `dataset/chunks.csv`
- `dataset/answers.csv` (15 questions, retrieved context, and generated answers)
- `chroma_db/` (created automatically)

The answers are generated from the retrieved PDF context. Always compare them with the PDF before submitting.
