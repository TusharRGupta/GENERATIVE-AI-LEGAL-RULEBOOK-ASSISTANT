import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs
def extract_text(pdfs):
    combined_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text

# Split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=9000, chunk_overlap=500)
    return text_splitter.split_text(text)

# Generate vector store
def create_vector_database(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_db.save_local("vector_index")

# Generate response
def generate_answer_chain():
    qa_prompt = """
    Use the given context to provide an accurate answer. If the answer isn't in the context, then provide answer to best of your knowlwdge. Provide the most detailed answer possible.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    llm_model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.4)
    prompt = PromptTemplate(template=qa_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm_model, chain_type="stuff", prompt=prompt)
    return chain

def reset_chat_history():
    st.session_state.conversation = [{"role": "assistant", "content": "Welcome! Upload your PDFs and ask a question."}]

def handle_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    relevant_docs = vector_db.similarity_search(user_question)

    response_chain = generate_answer_chain()
    response = response_chain(
        {"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
    
    return response

def main():
    st.set_page_config(page_title="PDF Knowledge Assistant", page_icon="📄")

    # Create a tabbed interface
    tabs = st.tabs(["Upload PDFs", "Chat", "Settings"])

    # PDF Upload Tab
    with tabs[0]:
        st.header("Upload and Process PDFs")
        st.write("Upload your PDFs here for processing.")
        uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_pdfs:
            with st.expander("Selected Files", expanded=True):
                for file in uploaded_pdfs:
                    st.write(file.name)
                
            if st.button("Process PDFs"):
                with st.spinner("Processing..."):
                    pdf_text = extract_text(uploaded_pdfs)
                    text_chunks = split_text_into_chunks(pdf_text)
                    create_vector_database(text_chunks)
                    st.success("Processing completed!")

    # Chat Tab
    with tabs[1]:
        st.header("Chat with Your Documents")
        st.write("Start a conversation with your uploaded PDFs.")
        
        # Chat history display
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Chat input
            prompt = st.chat_input("Ask a question...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                # Display response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = handle_user_question(prompt)
                        full_response = ''.join(response.get('output_text', []))
                        st.write(full_response)

                        # Add the assistant's response to the chat history
                        if response:
                            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Settings Tab
    with tabs[2]:
        st.header("Settings")
        st.write("Customize your experience.")
        if st.button("Clear Chat History"):
            reset_chat_history()
            st.success("Chat history cleared!")

if __name__ == "__main__":
    main()
