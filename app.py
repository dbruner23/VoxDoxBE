import os
import pickle
from langchain import hub
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from flask import Flask, request, jsonify, Response, session
from flask_cors import CORS
from dotenv import find_dotenv, load_dotenv
from functions import load_pdf, clear_documents, cited_answer
from helpers import prepare_whole_document_for_frontend, convert_ai_msg_to_json, convert_messages_to_dict
from werkzeug.utils import secure_filename

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
index_name = os.getenv("PINECONE_INDEX_NAME")

flask_app = Flask(__name__)
CORS(flask_app)
flask_app.secret_key = os.urandom(24) 

@flask_app.route("/process", methods=["POST"])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)
    file_type = request.form.get('fileType')
    
    if file_type not in {'csv', 'pdf', 'html', 'md'}:
        return jsonify({'error': 'Invalid file type'}), 400
    
    print(file_type)
    
    if (file_type == 'pdf'):
        vectorstore, pages =  load_pdf(filepath)
        pages_json = prepare_whole_document_for_frontend(pages)

    os.remove(filepath)
    return jsonify(pages_json), 200

@flask_app.route("/query", methods=["POST"])
def query():
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
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    if "chat_history" not in session:
        session["chat_history"] = []
        
    data = request.get_json()
    user_question = data.get("question")
    print(session["chat_history"])
    ai_msg = rag_chain_with_source.invoke(user_question)
    ai_msg = rag_chain_with_source.invoke(contextualized_question({"question": user_question, "chat_history": session["chat_history"]}))
    session["chat_history"].extend([convert_messages_to_dict(HumanMessage(content=user_question)), convert_messages_to_dict(ai_msg)])
    ai_msg_json = convert_ai_msg_to_json(ai_msg)
    
    return ai_msg_json, 200

@flask_app.route("/delete", methods=["GET"])
def delete_documents():
    clear_documents()
    return jsonify({"message": "All documents deleted"}), 200
    

if __name__ == "__main__":
    flask_app.run(debug=True, port=4000)