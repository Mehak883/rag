import os
from typing import Optional
from dotenv import load_dotenv
import chromadb
import openai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext,Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing")

openai.api_key = OPENAI_API_KEY
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)


class ChromaDBManager:
    def __init__(self, collection_name: str = "ps_vector"):
        self.client = chromadb.PersistentClient(path="./chroma_db")  # Local ChromaDB
        print("ChromaDB client initialized.")

        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created or loaded.")

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

    def store_embeddings(self, documents):
        if not documents:
            print("No valid documents found.")
            return

        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=embed_model
        )
        print(f"Successfully indexed {len(documents)} documents.")

    def query_db(self, query: str) -> Optional[str]:
        if not hasattr(self, 'index'):
            return "No documents indexed."

        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return response.response if response else "❓ No relevant information found."

def load_documents(file_path: str):
    """Loads a single file into LlamaIndex."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return documents
def main():
    print("----------- RAG System --------------")

    db_manager = ChromaDBManager()

    file_or_folder = input("Enter a pdf file path: ").strip()

    documents = load_documents(file_or_folder)
    if not documents:
        print("No valid PDF found.")
        return

    db_manager.store_embeddings(documents)

    while True:
        query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting RAG System. Bye!")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        context = db_manager.query_db(query)
        print("\nAnswer:", context)


if __name__ == "__main__":
    main()


# import os
# from typing import List, Optional
# from dotenv import load_dotenv
# import chromadb
# import openai
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.vector_stores.chroma import ChromaVectorStore

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("OpenAI API key is missing")

# # OpenAI API Key for OpenAI Client
# openai.api_key = OPENAI_API_KEY

# # Initialize OpenAI embedding model
# embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)



         
# class ChromaDBManager:
#     def __init__(self, collection_name: str = "ps_vector"):
#         # Initialize ChromaDB client
#         self.client = chromadb.PersistentClient(path="./chroma_db")
#         print("ChromaDB client initialized.")

#         # Create or load a collection
#         self.collection = self.client.get_or_create_collection(name=collection_name)
#         print(f"Collection '{collection_name}' created or loaded.")

#         # Initialize OpenAI embedding model
#         self.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)


#     def store_embeddings(self, documents):
#         if not documents:
#             print("No valid documents found.")
#             return

#         self.index = VectorStoreIndex.from_documents(
#             documents,
#             storage_context=self.storage_context,
#             embed_model=embed_model
#         )
#         print(f"Successfully indexed {len(documents)} documents.")

#     def query_db(self, query: str) -> Optional[str]:
#         if not self.index:
#             return "No documents indexed."

#         query_engine = self.index.as_query_engine()
#         response = query_engine.query(query)
#         return response.response if response else "❓ No relevant information found."


# class AnswerGenerator:
#     # Generates AI responses
#     @staticmethod
#     def generate_answer(context: str, query: str) -> str:
#         try:
#             client = openai.OpenAI(api_key=OPENAI_API_KEY)
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful AI assistant."},
#                     {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
#                 ],
#                 max_tokens=100,
#                 temperature=0.7,
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error: {e}")
#             return "Sorry, I encountered an error while generating a response."


# def main():
#     print("----------- RAG System --------------")

#     # Initialize Database
#     db_manager = ChromaDBManager()

#     # Load and process PDFs
#     pdf_folder = input("Enter the folder path containing PDFs: ").strip()
#     documents = SimpleDirectoryReader(pdf_folder).load_data()

#     if not documents:
#         print("No valid PDF found")
#         return

#     db_manager.store_embeddings(documents)

#     while True:
#         query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
#         if query.lower() == "exit":
#             print("Exiting RAG System. Bye!")
#             break

#         if not query:
#             print("Please enter a valid query.")
#             continue

#         context = db_manager.query_db(query)
#         answer = AnswerGenerator.generate_answer(context, query) if context else "No relevant information found."
#         print("\nAnswer:", answer)


# if __name__ == "__main__":
#     main()
