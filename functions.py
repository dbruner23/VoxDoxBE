import os
import json
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from helpers import convert_ai_msg_to_json

load_dotenv(find_dotenv())
index_name = os.getenv("PINECONE_INDEX_NAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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
    
def summarize_document2(user_question: str, splits_dicts: List[dict]):
    llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", anthropic_api_key=ANTHROPIC_API_KEY, temperature=0.5)
    
    # splits = [Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata']) for doc_dict in splits_dicts]
    
    summarize_system_prompt = """You are an assistant for summarization tasks. \
    Given the following document, please give a short summary of the content. \
        
    {docs}"""
    
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summarize_system_prompt),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),    
        ]
    )
    
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    
    splits_str = json.dumps(splits_dicts)
    
    summary = summarize_chain.invoke({"docs": splits_str, "question": user_question})
    
    return summary

def query_document(contextualized_question: str):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    
    if vectorstore is None:
        return {"error": "No documents loaded"}
    
    retriever =  vectorstore.as_retriever()
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{question}"),    
        ]
    )
    
    def format_docs(docs: List[Document]):
        formatted = [
            f"Source ID: {i}\nArticle Snippet: {doc.page_content}" for i, doc in enumerate(docs)
        ]
        return "\n\n" + "\n\n".join(formatted)
    
    llm_with_tool = llm.bind_tools(
        [cited_answer],
        tool_choice="cited_answer",
    )
    
    output_parser = JsonOutputKeyToolsParser(key_name="cited_answer", return_single=True)
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | qa_prompt
        | llm_with_tool
        | output_parser
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    ai_msg = rag_chain_with_source.invoke(contextualized_question)
    
    return ai_msg