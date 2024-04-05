import os
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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
import boto3
from botocore.exceptions import NoCredentialsError

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
index_name = os.getenv("PINECONE_INDEX_NAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORDHASH = os.getenv("APP_PASSWORDHASH")
MAX_CHAT_HISTORY_LENGTH = 20
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

flask_app = Flask(__name__)
auth = HTTPBasicAuth()
CORS(flask_app)
flask_app.secret_key = os.urandom(24)
session_id = os.urandom(24).hex() 

@auth.verify_password
def verify_password(username, password):
    if username == APP_USERNAME:
        password_hash = APP_PASSWORDHASH 
        if check_password_hash(password_hash, password):
            return True

tools = [
    {
        "type": "function",
        "function": {
            "name": "answer_queries_with_citations",
            "description": "useful for answering questions about article or document content.",
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
            "description": "use this only when very specifically asked for a summary of the document using the words 'summarize' or 'summary'.",
            "parameters": {},
            "required": [],
        }
    }
]

@flask_app.route("/auth", methods=["POST"])    
def authenticate():
    credentials = request.json
    username = credentials.get('username')
    password = credentials.get('password')

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    if verify_password(username, password):
        return jsonify({"message": "Authentication successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@flask_app.route("/docquery", methods=["POST"])
@auth.login_required
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
    # print(ai_function_call.additional_kwargs['tool_calls'][0]["function"]["name"])
    
    summary_dict = None
    answer_dict = None
    if "tool_calls" in ai_function_call.additional_kwargs and ai_function_call.additional_kwargs['tool_calls'][0]["function"]["name"] == "summarize_document":
        splits_dicts = load_splits_from_file(session_id)
        if not splits_dicts:
            return jsonify({"error": "No documents loaded"}, 400)
        else:
            summary = summarize_document2(user_question, splits_dicts)
            summary_dict = {"answer": summary}
            response = jsonify(summary_dict, 200)
    else:
        answer_with_citations = query_document(question_in_context)
        response = jsonify({"answer": convert_ai_msg_to_json(answer_with_citations)}, 200) 
    
    if summary_dict:    
        session["chat_history"].extend([convert_messages_to_dict(HumanMessage(content=user_question)), convert_messages_to_dict(summary_dict)])
    elif answer_dict:
        session["chat_history"].extend([convert_messages_to_dict(HumanMessage(content=user_question)), convert_messages_to_dict(answer_with_citations)]) 
    
    if len(session["chat_history"]) > MAX_CHAT_HISTORY_LENGTH:
        session["chat_history"] = session["chat_history"][1:]
    
    return response

@flask_app.route("/process", methods=["POST"])
@auth.login_required
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    session['current_filepath'] = filepath
    file.save(filepath) 
    file_type = request.form.get('fileType')
    
    if file_type not in {'pdf'}:
        return jsonify({'error': 'Invalid file type'}), 400
    
    # try:
    #     s3_client.upload_fileobj(file, S3_BUCKET_NAME, filename)
    #     print(filename)
    #     file_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
    #     session['current_file_url'] = file_url
    # except NoCredentialsError:
    #     return jsonify({'error': 'AWS credentials not available'}), 500
    
    print(file_type)
    
    if (file_type == 'pdf'):
        vectorstore, pages, splits =  load_pdf(filepath)
        save_splits_to_file(serialize_documents(splits), session_id)
        pages_dicts = serialize_documents(pages)

    return jsonify(pages_dicts), 200    

@flask_app.route("/delete", methods=["GET"])
@auth.login_required
def delete_documents():
    if "current_filepath" in session:
        del session["current_filepath"]
    clear_documents()
    return jsonify({"message": "All documents deleted"}), 200
    

if __name__ == "__main__":
    flask_app.run(debug=True, port=4000)
