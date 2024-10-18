from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import openai
import langchain
import requests
import pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import environ
from langchain_community.chat_models import ChatOpenAI
from pypdf import PdfReader
from langchain.schema import Document
import os
from pinecone import Pinecone, ServerlessSpec
env = environ.Env()
environ.Env.read_env()

os.environ['PINECONE_API_KEY'] = '2ce5d867-ea99-4eca-a310-48cdc9cd39b0'

# Initialize Openai Embeddings
embedding = OpenAIEmbeddings(api_key='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')
index_name = "medical-bot"

# Function to read documents from a directory
def read_doc(file_url):
    print(f"Downloading file from URL: {file_url}")
        
        # Download the file
    response = requests.get(file_url)
    response.raise_for_status()
    
    pdf_content = response.content
    print(f"Downloaded {len(pdf_content)} bytes")
    
    pdf_file = BytesIO(pdf_content)
    reader = PdfReader(pdf_file)
    
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append(Document(page_content=text, metadata={'page': i+1}))
    
    print(f"Loaded {len(documents)} pages.")
    return documents

# Function to chunk documents with metadata
def chunk_data(docs, chunk_size=2000, chunk_overlap=50, document_id=None):
    try:
        print('enter')
        docs_with_page_content = [Document(page_content=doc) for doc in docs]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs_with_page_content)
        for chunk in chunks:
            chunk.metadata = {
                'document_id': document_id
            }
        print(f"Chunked into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(e)
        raise Exception
    
# Function to create the vector database
def create_VDB(file_url, document_id=''):
    documents = csv_to_textual_representation(file_url)
    print('outside csv')
    chunked_doc = chunk_data(docs=documents, document_id=document_id)
    try:
        print(embedding)
        print(chunked_doc)
        print(index_name)
        try:
            index = PineconeVectorStore.from_documents(chunked_doc, index_name=index_name, embedding=embedding)
        except Exception as e:
            print(e)
            raise Exception 
        print("Index created and populated with documents.")
        return index
    except Exception as e:
        print(e)
        raise Exception



# Function to create or load the index
def create_or_load_index():
    try:
        index = PineconeVectorStore(index_name=index_name, embedding=embedding)
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Index not found, creating a new one. Error: {e}")
        index = create_VDB()
    return index

# Initialize or load the index once
# VDB_INDEX = create_or_load_index()

# Function to retrieve query results from the vector database with metadata filtering
def retrieve_query(query, document_id=None, k=10):
    index = create_or_load_index()
    filters = {}
    if document_id:
        filters['document_id'] = document_id

    matching_similarity = index.similarity_search(query, k=k, filter=filters)
    
    if not matching_similarity:
        print("No similar documents found.")
    else:
        for doc in matching_similarity: # Print first 100 characters
            print(f"Metadata: {doc.metadata}")
    return matching_similarity

# Function to retrieve answers based on the query
def retrieve_answers(query, document_id=None):
    llm = OpenAI(temperature=0 ,api_key = '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')
    similar_docs = retrieve_query(query, document_id=document_id)

    # Extract text content from the retrieved documents
    context_texts = [doc.page_content for doc in similar_docs]
    context = "\n".join(context_texts)

    custom_prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template="Here is the context provided: ```{context}```\nBased on this context, please answer the following question: ```{question}```\n Give the response according to information in the provided context\n response should be in points also avoid using'/n'/nIf the answer is not found in the context, simply respond with 'I don't know'."
    )
    chain = LLMChain(llm=llm, prompt=custom_prompt_template)

    # Manually construct the prompt
    raw_prompt = custom_prompt_template.format(context=context, question=query)

    # Print the raw prompt
    print("Raw Prompt:", raw_prompt)

    result = chain.run(context=context, question=query)
    return result


def get_chunk_ids_to_delete( document_id=None):
    index = create_or_load_index()
    filters = {}
    if document_id:
        filters['document_id'] = document_id

    matching_chunks = index.similarity_search(query="", k=1000, filter=filters)  # Use a dummy query and a large k value to retrieve all matches
    chunk_ids = [chunk.id for chunk in matching_chunks]
    return chunk_ids

def delete_chunks_by_ids(chunk_ids):
    if chunk_ids:
        index = create_or_load_index()
        index.delete(ids=chunk_ids)
        print(f"Deleted {len(chunk_ids)} chunks.")
    else:
        print("No chunks found to delete.")


import csv
def csv_to_textual_representation(file_path, max_rows=15000):
    """
    Converts up to the first 10,000 rows of a CSV file into a textual representation.
    
    Args:
        file_path (str): Path to the CSV file.
        max_rows (int, optional): Maximum number of rows to process. Default is 10,000.
    
    Returns:
        list: A list of strings where each string represents a row in the CSV.
    """
    textual_data = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Get the header row
        
        for i, row in enumerate(reader):
            if i >= max_rows:  # Stop reading after 10,000 rows
                break
            # Convert each row into a textual representation
            row_text = ', '.join([f"{headers[j]}: {value}" for j, value in enumerate(row)])
            textual_data.append(row_text)
    
    return textual_data