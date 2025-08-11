# Talk With Docs RAG Memory

🔗 **Live App:** [Talk With Docs RAG Memory](https://talkwithdocsragmemory.streamlit.app/)

---

## 📌 Overview

**Talk With Docs RAG Memory** is a Streamlit-powered conversational Retrieval-Augmented Generation (RAG) application that allows users to upload and chat with PDF documents.  
Built with **Groq LLM** for language model capabilities and **Hugging Face embeddings**, this app supports multi-file PDF uploads, generates embeddings, and answers user queries with memory of past interactions to provide context-rich responses.

---

## 🚀 Features

- 📄 Upload multiple PDF files for analysis.
- 🧠 Conversational memory for context-aware responses.
- 🔍 Generate and use embeddings from uploaded documents.
- 🤖 Integration with **Groq LLM** for advanced natural language responses.
- 🎯 Clean and responsive UI using **Streamlit**.

---

## 🗂 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, Groq LLM
- **Embeddings:** Hugging Face embeddings
- **RAG & Memory:** LangChain agents

---

## 📦 Installation & Setup

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/akhil-droid/Talk_With_Docs_RAG_Memory.git
cd Talk_With_Docs_RAG_Memory
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Configure Environment Variables**
- Copy `.env.sample` to `.env`
- Add your credentials such as `GROQ_API_KEY`.

### **Step 4: Run the Application**
```bash
streamlit run app.py
```

Once running, open the provided local URL (e.g., `http://localhost:8501`) to access the app.

---

## ▶️ Usage

1. Upload one or more PDF documents.
2. Ask your questions related to the uploaded content.
3. The app processes the PDFs, generates embeddings, queries the LLM, and responds with context-aware answers.

---

## 💡 Example Queries

- "Summarize the main findings in the uploaded research papers."
- "Compare the key sections across all uploaded PDFs."
- "What are the concluding remarks in the last document I uploaded?"

---

## 📂 Project Structure

```
Talk_With_Docs_RAG_Memory/
│
├── app.py            # Main Streamlit app
├── requirements.txt  # Required Python packages
├── .env.sample       # Template for environment variables
├── LICENSE           # Project license
└── README.md         # Documentation
```

---

## 📝 Notes

- Python **3.8+** is recommended.
- Ensure valid API keys are provided in the `.env` file.
- Extendable to other file types, embedding methods, and LLMs.

---

## 📜 License

Licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgments

- [Groq LLM](https://www.groq.com/)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)

---

💡 *Chat smarter with your documents!* 📚🤖
