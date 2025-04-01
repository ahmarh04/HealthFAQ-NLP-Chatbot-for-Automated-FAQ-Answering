# HealthFAQ-NLP-Chatbot-for-Automated-FAQ-Answering ✨

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Sentence--Transformers-orange)](https://www.sbert.net/)
[![Interface](https://img.shields.io/badge/Interface-ipywidgets-purple)](https://ipywidgets.readthedocs.io/en/latest/)

---

## 📄 Overview

IntelliFAQ is an AI-powered chatbot designed to understand user questions and provide relevant answers from a predefined Frequently Asked Questions (FAQ) database. It leverages Natural Language Processing (NLP) techniques, specifically semantic similarity search using sentence embeddings, to match user queries with the most relevant FAQ entry.

This project uses a curated set of FAQs regarding COVID-19 vaccination guidelines (sourced from Ontario public health information) as its knowledge base, demonstrating its capability in a real-world information retrieval scenario.

---

## ⭐ Features

*   **Semantic Query Understanding:** Goes beyond keyword matching to understand the *meaning* behind user questions.
*   **Relevant Answer Retrieval:** Identifies the best-matching FAQ entry from the knowledge base.
*   **Source & Confidence:** Provides the source of the information and a confidence score for transparency.
*   **Fallback Mechanism:** Gracefully handles queries outside the scope of the FAQ database.
*   **Interactive Interface:** Includes an `ipywidgets`-based chat interface directly within the Jupyter Notebook for easy interaction and demonstration.

---

## 🧠 How it Works

1.  **FAQ Embedding:** The curated FAQ questions are pre-processed and encoded into high-dimensional vector embeddings using the `all-mpnet-base-v2` Sentence Transformer model. These embeddings capture the semantic meaning of the questions and are stored for efficient comparison.
2.  **User Query:** The user inputs a question via the interactive interface.
3.  **Query Embedding:** The user's raw query is encoded into a vector embedding using the same Sentence Transformer model.
4.  **Similarity Calculation:** Cosine Similarity is computed between the user's query embedding and all pre-computed FAQ question embeddings.
5.  **Best Match Identification:** The FAQ entry with the highest cosine similarity score to the user's query is identified.
6.  **Thresholding & Response:**
    *   If the highest similarity score meets or exceeds a predefined confidence threshold (e.g., 0.6-0.7), the corresponding answer, source, and confidence score are returned to the user.
    *   If the score is below the threshold, a default message is returned indicating the information might not be available in the current knowledge base.

---

## 🚀 Technology Stack

*   **Language:** Python 3
*   **Core NLP/Embeddings:** `sentence-transformers` (specifically `all-mpnet-base-v2` model)
*   **Deep Learning Framework:** `PyTorch` (backend for Sentence Transformers)
*   **Text Processing (Initial Setup):** `NLTK` (for downloading models/data like 'punkt', 'stopwords', 'wordnet')
*   **Interactive UI:** `ipywidgets` (within Jupyter/Kaggle environment)
*   **Data Handling (Optional):** `pandas`

---

## 📊 Dataset

The knowledge base consists of a set of Question/Answer pairs related to COVID-19 vaccination, sourced from official Ontario Ministry of Health and Public Health Ontario guidelines (circa 2023). The specific questions, answers, and sources are defined within the notebook.

---

## 🛠️ Setup & Installation

1.  **Clone the repository (Optional):**
    ```bash
    git clone https://github.com/YourGitHubUsername/IntelliFAQ.git
    cd IntelliFAQ
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install sentence-transformers torch nltk ipywidgets pandas numpy
    ```
    *(**Note:** Ensure you have a compatible version of PyTorch installed, potentially with CUDA support if using a GPU.)*

3.  **Download NLTK data:** Run the following Python commands once (or ensure the relevant cells in the notebook are run):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```

---

## ⚙️ Usage

The primary way to use and interact with IntelliFAQ is through the Jupyter Notebook:

1.  Open the notebook `notebook6af4da59da.ipynb` in a compatible environment (like Jupyter Lab, VS Code with Python extension, or Kaggle).
2.  Run the cells sequentially from top to bottom. This will:
    *   Install necessary libraries (if not already done).
    *   Download required NLTK data.
    *   Load the Sentence Transformer model.
    *   Define and embed the FAQ data.
    *   Define the `vaccine_chatbot` function.
    *   Launch the interactive `ipywidgets` chat interface at the bottom of the notebook.
3.  Type your questions into the input box in the chat interface and press Enter or click "Send".

---

## 🔮 Future Work

*   Integrate more advanced transformer models (e.g., fine-tuned BERT, GPT variants) for potentially improved nuance detection.
*   Expand the FAQ database with more topics or domains.
*   Implement a mechanism for dynamic knowledge updates.
*   Develop a standalone web application interface (e.g., using Flask/Django).
*   Add multilingual support.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You'll need to add a LICENSE file, MIT is a common choice).

---
