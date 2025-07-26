# 🧠 Retrieval-Augmented Generation (RAG) with Mistral & FAISS

This repository demonstrates a complete pipeline for building a **Retrieval-Augmented Generation (RAG)** system using the **Mistral 7B** language model and **FAISS** for scalable vector search. It integrates document ingestion, chunking, semantic embedding, vector indexing, and context-aware querying using a powerful open-source LLM.

---

## 🚀 Overview

RAG enhances LLM performance by incorporating external, domain-specific knowledge at inference time. This notebook implements:

- 📄 Multi-format document loading (PDF, text)
- ✂️ Context-aware chunking with LangChain
- 🔎 Semantic embedding generation using Hugging Face models
- ⚡ Fast and efficient vector search with FAISS
- 🤖 Query-time retrieval and generation using Mistral 7B

---

## 🗂️ Project Structure

```bash
.
├── building-rag-using-mistral-faiss-v2.ipynb   # Main notebook
├── data/                                       # Input documents
│   └── sample_docs/
├── faiss_index/                                # Persisted FAISS index
├── requirements.txt                            # Python dependencies
└── README.md
````

---

## ⚙️ Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/rag-mistral-faiss.git
   cd rag-mistral-faiss
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download or add documents** to `data/sample_docs/`.

---

## 📌 Key Components

| Component       | Description                                                 |
| --------------- | ----------------------------------------------------------- |
| Document Loader | Loads PDFs/text files into raw text using LangChain         |
| Text Splitter   | Splits content into semantically coherent chunks            |
| Embedding Model | Generates vector representations (HuggingFace Transformers) |
| FAISS Index     | Stores and retrieves vectors with similarity search         |
| RAG Generator   | Uses Mistral 7B for context-augmented answer generation     |

---

## 🧪 Usage

Run the Jupyter notebook step-by-step:

1. **Ingest & chunk** your documents.
2. **Generate embeddings** and build FAISS index.
3. **Save/load index** to/from disk.
4. **Run inference** with any natural language query.

Example:

```python
query = "What are the benefits of using RAG over standard LLMs?"
```

---

## 💡 Why Mistral + FAISS?

* **Mistral 7B**: High-performance, open-source LLM optimized for efficiency and accuracy.
* **FAISS**: Fast Approximate Nearest Neighbors Search—essential for scalable retrieval.
* **LangChain**: Modular pipeline integration for NLP tasks and document intelligence.

---

## 📁 Saving & Reloading the Index

```python
# Save index
vectorstore.save_local("faiss_index/")

# Reload index
vectorstore = FAISS.load_local("faiss_index/", embeddings)
```

---

## 🔐 Notes

* Add API keys or authentication (if using external LLM APIs).
* For local inference, integrate `llama-cpp-python` or `vllm` for fast performance.
* GPU acceleration recommended for large-scale embeddings and inference.

---

## 🧠 Acknowledgements

* [Mistral AI](https://mistral.ai) for the model
* [LangChain](https://www.langchain.com) for orchestration
* [Facebook FAISS](https://github.com/facebookresearch/faiss) for vector search

---

## 📜 License

This project is licensed for research and educational use only. Commercial use requires appropriate licensing from respective model and tool providers.

---

## ✨ Future Enhancements

* 🧾 Web UI (e.g., Streamlit or Gradio)
* 🏃 Real-time document ingestion and indexing
* 🧠 Multi-document summarization
* 📚 Support for additional formats (DOCX, HTML, etc.)


