import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# -----------------------------------------------
# Loading environmenrt variables

from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------
# Directory Paths

current_dir_path = os.path.dirname(os.path.abspath(__file__))
# print(current_dir_path)
books_dir_path = os.path.join(current_dir_path, "Books")
# print(books_dir_path)
database_dir_path = os.path.join(current_dir_path, "Database")
# print(database_dir_path)
metadata_dir = os.path.join(database_dir_path, "Chroma-db-with-Metadata")
# print(metadata_dir)

# -----------------------------------------------
# Checking is Chroma vector database exist

if not os.path.exists(metadata_dir):
    print("Chroma database does not exist. Initialzing it...")

    # -----------------------------------------------
    # Checking if the Books directory exist

    if not os.path.exists(books_dir_path):
        raise FileNotFoundError(f"The directory {books_dir_path} does not exist.")
    
    # -----------------------------------------------
    # Get all book text files

    book_txt_files = [file for file in os.listdir(books_dir_path) if file.endswith(".txt")]
    # print(len(book_txt_files))

    # -----------------------------------------------
    # Read the text content from each file with metadata

    documents = []

    for book_file in book_txt_files:
        file_path = os.path.join(books_dir_path, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Adding metadata to each document
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # -----------------------------------------------    
    # Splitting the documents into chunks

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # -----------------------------------------------
    # Length of the split document

    print(f"Number of Chunks : {len(docs)}")

    # -----------------------------------------------
    # Let's create document embeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # -----------------------------------------------
    # Create a vector store to save the document and embeddings

    db = Chroma.from_documents(docs, embeddings, persist_directory=metadata_dir)

else:
    print("Vector store already exist.")

# -----------------------------------------------