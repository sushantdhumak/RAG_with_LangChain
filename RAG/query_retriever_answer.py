import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

# -----------------------------------------------
# Directory Paths

current_dir_path = os.path.dirname(os.path.abspath(__file__))
database_dir_path = os.path.join(current_dir_path, "Database")
metadata_dir_path = os.path.join(database_dir_path, "Chroma-db-with-Metadata")

# -----------------------------------------------
# Embedding Model

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------------------------------
# Load the existing vector store

db = Chroma(persist_directory=metadata_dir_path, embedding_function=embeddings)

# -----------------------------------------------
# User Question

# query = "Who is Dumbledore?"
query = "Who opened the chamber of secrets?"
# query = "What is the name of GOD?"
# query = "Why people were worshiping a calf?"

# -----------------------------------------------
# Retriving relevant documents from vector db

retriever = db.as_retriever(
    # search_type="similarity_score_threshold", 
    # search_kwargs = {"k": 5, "score_threshold": 0.2}
    search_type="similarity", 
    search_kwargs = {"k": 5}
    )

relevant_docs = retriever.invoke(query)

# -----------------------------------------------
# Display the relevant documents

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} :\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")

# -----------------------------------------------
# Let's fine-tune the input based on relevant docs

final_input = (
    "Use thses documents to answer the given query : "
    + query
    + "\n\n Relevant Documents :\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\n Please provide answer based only on the provided documents."
    + "\n If the answer is not found in the document, just respond 'Answer is not in my knowledge store'."
)

# -----------------------------------------------
# Use the ChatOpenAI model

model = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------------------------
# Message to the model

message = [
    AIMessage(content="You are a helpful assistant."),
    HumanMessage(content=final_input),
]

# -----------------------------------------------
# Invoke the model

response = model.invoke(message)

# -----------------------------------------------
# Display the results

print("Full Response : \n")
print(response)

print("\nOnly Response Content : \n")
print(response.content)

# -----------------------------------------------