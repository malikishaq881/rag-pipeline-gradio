# RAG Pipeline — Retrieval-Augmented Generation

End-to-end RAG system built in Google Colab using FAISS, SentenceTransformers, TinyLlama, and Gradio.

## Quick start
1. Open `rag_pipeline.ipynb` in Google Colab (T4 GPU recommended)
2. Run all cells top-to-bottom
3. Upload your `.txt` files when prompted in Cell 3
4. The Gradio interface launches at Cell 11 with a public share link

## Features
- Document loading, chunking (400 chars / 80 overlap), and FAISS indexing
- Semantic search with `all-MiniLM-L6-v2` (384-dim embeddings)
- Answer generation with `TinyLlama-1.1B-Chat` — loaded once at startup
- Displays answer + sources (filename, chunk, similarity score) for every query
- Warning message when no sources pass the similarity threshold
- Gradio sliders for TOP_K (1–10) and MIN_SCORE (0–1)

## Stack
`sentence-transformers` · `faiss-cpu` · `transformers` · `gradio` · `torch`
