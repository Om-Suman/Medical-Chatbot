import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

HF_TOKEN = os.environ.get("HF_TOKEN") 
client = InferenceClient(token=HF_TOKEN)

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        client=InferenceClient(token=HF_TOKEN),
        temperature=0.5,
        max_new_tokens=512 
    )
    return llm

def main():
    st.title("MediBot - Your Medical Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your medical question here:")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            vector_store = get_vectorstore()
            if vector_store is None:
                raise ValueError("Vector store is not loaded properly.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGING_FACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response['result']
            source_docs = response.get('source_documents', [])

            result_to_show = result + "\n\nSOURCE DOCUMENTS:\n" + "\n\n".join(
                 [f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content.strip()}" for doc in source_docs])

            st.chat_message('assistant').write(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
