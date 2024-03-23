import os
import json
import pickle
from getpass import getpass
from dataclasses import asdict
from langchain import hub
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_anthropic import AnthropicLLM
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from flask import Flask, request, jsonify, Response, session
from flask_cors import CORS
from dotenv import find_dotenv, load_dotenv
from functions import load_pdf, clear_documents, cited_answer, summarize_document2, query_document
from helpers import (
    serialize_documents,
    convert_ai_msg_to_json,
    convert_messages_to_dict,
    save_splits_to_file,
    load_splits_from_file,
    delete_splits_file
)
from werkzeug.utils import secure_filename

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
index_name = os.getenv("PINECONE_INDEX_NAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MAX_CHAT_HISTORY_LENGTH = 20

flask_app = Flask(__name__)
CORS(flask_app)
flask_app.secret_key = os.urandom(24)
session_id = os.urandom(24).hex() 

tools = [
    {
        "type": "function",
        "function": {
            "name": "answer_queries_with_citations",
            "description": "useful for when you need to answer a particular question with citations.",
            "parameters": {},
            "required": [],
        }
    },
    {
        "type": "function",
        "function": {
            "name": "miscellaneous_question_answerer",
            "description": "useful for answering miscellaneous questions.",
            "parameters": {},
            "required": [],
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "useful only when specifically asked for a document summary.",
            "parameters": {},
            "required": [],
        }
    }
]

@flask_app.route("/test", methods=["POST"])
def test():
    data = request.get_json()
    user_question = data.get("question")
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)
    llm_with_tools = llm.bind_tools(
        tools=tools,
        tool_choice="auto",
    )
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    contextualize_q_chain =  contextualize_q_prompt | llm | StrOutputParser()
    
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            contextualized_question_output = contextualize_q_chain.invoke(input)
            print("CONTEXTUALIZED OUTPUT:", contextualized_question_output)
            return contextualized_question_output
        else:
            return input["question"]
        
    if "chat_history" not in session:
        session["chat_history"] = []
        
    question_in_context = contextualized_question({"question": user_question, "chat_history": session["chat_history"]})

    ai_function_call = llm_with_tools.invoke(question_in_context)
    
    if ai_function_call.additional_kwargs['tool_calls'] and ai_function_call.additional_kwargs['tool_calls'][0]["function"]["name"] == "summarize_document":
        splits_dicts = load_splits_from_file(session_id)
        if not splits_dicts:
            response = jsonify({"error": "No documents loaded"}), 400
        else:
            summary = summarize_document2(user_question, splits_dicts)
            response = jsonify({"answer": summary}), 200
    else:
        answer_with_citations = query_document(question_in_context)
        response = jsonify({"answer": answer_with_citations}), 200 
    
    # if response["answer"]:    
    #     print("add answer to chat history")
        # session["chat_history"].extend([convert_messages_to_dict(HumanMessage(content=user_question)), convert_messages_to_dict()])
    
    if len(session["chat_history"]) > MAX_CHAT_HISTORY_LENGTH:
        session["chat_history"] = session["chat_history"][1:]
    
    return response

@flask_app.route("/process", methods=["POST"])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    session['current_filepath'] = filepath
    file.save(filepath) 
    file_type = request.form.get('fileType')
    
    if file_type not in {'csv', 'pdf', 'html', 'md'}:
        return jsonify({'error': 'Invalid file type'}), 400
    
    print(file_type)
    
    if (file_type == 'pdf'):
        vectorstore, pages, splits =  load_pdf(filepath)
        save_splits_to_file(serialize_documents(splits), session_id)
        pages_dicts = serialize_documents(pages)

    return jsonify(pages_dicts), 200

@flask_app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_question = data.get("question")
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    
    if vectorstore is None:
        return jsonify({"error": "No documents loaded"}), 400
    
    retriever =  vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    contextualize_q_chain =  contextualize_q_prompt | llm | StrOutputParser()
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),    
        ]
    )
    
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            contextualized_question_output = contextualize_q_chain.invoke(input)
            print("CONTEXTUALIZED OUTPUT:", contextualized_question_output)
            return contextualized_question_output
        else:
            return input["question"]
    
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
    
    if "chat_history" not in session:
        session["chat_history"] = []
        
    # ai_msg = rag_chain_with_source.invoke(user_question)
    ai_msg = rag_chain_with_source.invoke(contextualized_question({"question": user_question, "chat_history": session["chat_history"]}))
    session["chat_history"].extend([convert_messages_to_dict(HumanMessage(content=user_question)), convert_messages_to_dict(ai_msg)])
    print(ai_msg)
    ai_msg_json = convert_ai_msg_to_json(ai_msg)
    
    return ai_msg_json, 200

@flask_app.route("/summarise", methods=["POST"])
def summarize_document():
    data = request.get_json()
    user_question = data.get("question")

    llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", anthropic_api_key=ANTHROPIC_API_KEY, temperature=0.5)
    
    splits_dicts = load_splits_from_file(session_id)
    print(splits_dicts)
    if not splits_dicts:
        return jsonify({"error": "No documents loaded"}), 400
    splits = [Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata']) for doc_dict in splits_dicts]
    
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
    print(summary)
    
    return jsonify({"summary": summary}), 200
    
    

@flask_app.route("/delete", methods=["GET"])
def delete_documents():
    if "current_filepath" in session:
        del session["current_filepath"]
    # delete_splits_file(session_id)
    clear_documents()
    return jsonify({"message": "All documents deleted"}), 200
    

if __name__ == "__main__":
    flask_app.run(debug=True, port=4000)
    
    
# map_template = """The following is a set of documents
    # {docs}
    # Based on this list of docs, please identify the main themes 
    # Helpful Answer:"""
    # map_prompt = PromptTemplate.from_template(map_template)
    # map_chain = map_prompt | llm 
    
    # reduce_template = """The following is set of summaries:
    # {docs}
    # Take these and distill it into a final, consolidated summary of the main themes. 
    # Helpful Answer:"""
    # reduce_prompt = PromptTemplate.from_template(reduce_template)
    
    # reduce_chain = reduce_prompt | llm
    
    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=reduce_chain, document_variable_name="docs"
    # )
    
    # reduce_documents_chain = ReduceDocumentsChain(
    #     combine_documents_chain=combine_documents_chain,
    #     collapse_documents_chain=combine_documents_chain,
    #     token_max=4000,
    # )
    
    # map_reduce_chain = MapReduceDocumentsChain(
    #     llm_chain=map_chain,
    #     reduce_documents_chain=reduce_documents_chain,
    #     document_variable_name="docs",
    #     return_intermediate_steps=False,
    # )
    
    # summary = map_reduce_chain.run(splits)

    # chain = load_summarize_chain(
    #     llm=llm,
    #     chain_type="refine",
    #     input_key="input_documents",
    #     output_key="output_text"
    # )
    # summary = chain.run(splits)
    # print(summary)