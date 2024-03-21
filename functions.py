import os
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv(find_dotenv())
index_name = os.getenv("PINECONE_INDEX_NAME")

embeddings = OpenAIEmbeddings()

def load_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    vectorstore = PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
    
    return vectorstore, pages, splits

def clear_documents():
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    vectorstore.delete(delete_all=True)
    return {"message": "All documents deleted"}

class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )