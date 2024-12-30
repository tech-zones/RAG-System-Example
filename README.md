# RAG-based PDF Query System

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload multiple PDF files, extract and preprocess the text, and then query the contents of those PDFs using OpenAI's GPT-3.5-turbo model.

### Key Components and Technologies Used

- **Streamlit:** For building an interactive web application.
- **pdfplumber:** For extracting text from PDF files.
- **NLTK:** For text preprocessing tasks such as tokenization.
- **Scikit-learn:** For TF-IDF vectorization and text retrieval.
- **OpenAI GPT-3.5-turbo:** For generating context-aware responses to user queries.

