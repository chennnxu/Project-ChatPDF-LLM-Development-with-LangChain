import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())  # read local .env file

# Get the API keys from environment variables
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
baichuan_api_key = os.getenv('BAICHUAN_AUTH_TOKEN')

# Check if the API keys are available
if not deepseek_api_key:
    raise ValueError("DeepSeek API key not found in environment variables.")
if not baichuan_api_key:
    raise ValueError("Baichuan API key not found in environment variables.")

# Import PyPDFLoader from langchain_community
from langchain_community.document_loaders import PyPDFLoader

# Specify the path to the PDF file
file_path = "book/1.pdf"

# Create a PyPDFLoader instance
loader = PyPDFLoader(file_path)
# Load the PDF document
docs = loader.load()
print(f"Number of pages in the document: {len(docs)}")

# Print the first 100 characters of the content from the first page
print(docs[0].page_content[0:100])
# Print the metadata of the first page
print(docs[0].metadata)

# Import ChatOpenAI from langchain_openai
from langchain_openai import ChatOpenAI

# Initialize a ChatOpenAI instance with deepseek model
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=deepseek_api_key,
    openai_api_base="https://api.deepseek.com",
    max_tokens=1024
)

from langchain_chroma import Chroma
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Using a text splitter, split the loaded documents into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# Load them into a vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=BaichuanTextEmbeddings(baichuan_api_key=baichuan_api_key))
# Create a retriever from the vector store for use in our RAG chain
retriever = vectorstore.as_retriever()

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def ask_question(question):
    results = rag_chain.invoke({"input": question})
    print(f"Question: {question}")
    print(f"Answer: {results['answer']}")
    print(f"Source: {results['context'][0].metadata}")
    print("---")

# Example usage
ask_question("桃园三结义都有谁?")
ask_question("刘备的武器是什么?")