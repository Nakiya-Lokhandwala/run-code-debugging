# Python Code Debugging Assistant

Lightweight Streamlit app that combines a Retrieval-Augmented Generation (RAG) knowledge base with an LLM to debug or explain Python code. Uses a FAISS index over StackOverflow Q&A, a local SentenceTransformer embedder, and the Google Gemini (GenAI) client for generation.

Key files
- [app.py](app.py) — main Streamlit app and core logic (see functions: [`load_assets`](app.py), [`execute_code`](app.py), [`retrieve_similar`](app.py), [`generate_fix`](app.py), [`debug_code`](app.py))
- [requirements.txt](requirements.txt) — Python dependencies
- [.env](.env) — local environment variables (not checked into repo)
- [.gitignore](.gitignore)

Deployment & data
- [deployment/data/cleaned_stackoverflow.csv](deployment/data/cleaned_stackoverflow.csv) — cleaned Q&A dataset used for retrieval
- [deployment/data/faiss_index.index](deployment/data/faiss_index.index) — FAISS index for vector search
- [deployment/models/sentence_embedder/](deployment/models/sentence_embedder/) — local SentenceTransformer model files
- [deployment/stackoverflow_dataset/](deployment/stackoverflow_dataset/) — original dataset CSVs (Questions/Answers/Tags)

Features
- Executes user-supplied Python code in an isolated temporary file and captures stdout/stderr via [`execute_code`](app.py).
- Retrieves top-k similar StackOverflow Q&A from FAISS using the local embedder via [`retrieve_similar`](app.py).
- Generates a structured debugging response from the Gemini model using [`generate_fix`](app.py).
- Combines execution output, retrieved context, and LLM generation in the main flow implemented in [`debug_code`](app.py).

Quickstart

1. Clone the repo and create a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
