from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# step1 : Load raw pdf 

Data_Path = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, 
                             glob="**/*.pdf", 
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=Data_Path)
#print(f"Loaded {len(documents)} pages from PDF file.")

# step2 : Split the pdf into chunks 
def createChunks(extracted_data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks = createChunks(documents)
#print("Length of text chunks:", len(text_chunks))

# step3 : Create vector embeddings for each chunk

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()


# step 4 : Store the embeddings in a Faiss
DB_FAISS_Path = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_Path)