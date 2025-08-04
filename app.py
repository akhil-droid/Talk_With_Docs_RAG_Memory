## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"]= os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]= os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"]= os.getenv("LANGSMITH_TRACING")

# Emedding
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.set_page_config(page_title="Conversational RAG with PDFs", layout="wide")
st.title("Conversational RAG With PDF Uploads and Chat History")
st.markdown("Upload your PDFs and ask questions interactively.")

## chat interface

## Input the Groq API Key
api_key=st.text_input("Enter your **Groq API key**:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    # Session id
    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'history_store' not in st.session_state:
        st.session_state.history_store={}

    uploaded_files=st.file_uploader("Upload PDF files",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        with st.spinner("Processing documents..."):
            documents=[]
            for uploaded_file in uploaded_files:
                temppdf=f"./temp.pdf"
                with open(temppdf,"wb") as file:
                    file.write(uploaded_file.getvalue())
                    file_name=uploaded_file.name

                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                documents.extend(docs)

            # Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()    

            contextualize_q_system_prompt=(
                "Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
            
            history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

            ## Answer question prompt
            system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
            qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
            
            question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
            rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

            def get_session_history(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.history_store:
                    st.session_state.history_store[session_id]=ChatMessageHistory()
                return st.session_state.history_store[session_id]
            
            conversational_rag_chain=RunnableWithMessageHistory(
                rag_chain,get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            st.success("✅ PDF processed. You can now ask questions.")

            user_input = st.text_input("Your question:")
            if user_input:
                session_history=get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id":session_id}
                    },  # constructs a key "abc123" in `store`.
                )
                st.markdown("### 🤖 Assistant:")
                st.write(response['answer'])

                st.markdown("#### 📜 Chat History")
                for msg in get_session_history(session_id).messages:
                    st.markdown(f"**{msg.type.title()}:** {msg.content}")
else:
    st.warning("Please enter your Groq API key to continue.")










