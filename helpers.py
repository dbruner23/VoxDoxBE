import json
import os
from langchain_core.messages import HumanMessage
import html
import re

def serialize_documents(documents):
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
            # "source": doc.metadata['source'],
            # "pageNumber": doc.metadata['page']
        } 
        for doc in documents
    ]
    
def normalize_text(text):
    """Normalizes text by decoding HTML entities, fixing common replacements, and removing extra spaces."""

    # Decode HTML entities
    text = html.unescape(text)

    # Fix common replacements (ti -> fi, etc.) 
    text = re.sub(r'7', 'ti', text)  # Replace '7' with 'ti'
    
    text = re.sub(r'\ufb01' 'fi', text)

    # Remove extra spaces (optional)
    text = re.sub(r'\s+', ' ', text).strip()  

    return text
    
def convert_ai_msg_to_json(data):
    json_data = {
        "context": [],
        "question": data["question"],
        "answer": data["answer"]
    }

    for doc in data["context"]:
        json_doc = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        json_data["context"].append(json_doc)

    return json.dumps(json_data)


def convert_messages_to_dict(message):
    if isinstance(message, HumanMessage):
        return {"content": message.content, "role": "human"}
    else:
        print("MESSAGE: ", message)
        return {"content": message["answer"], "role": "ai"}
    
def save_splits_to_file(splits, session_id):
    filepath = f'/tmp/splits_{session_id}.json'
    print("SAVE FILEPATH: ", filepath)
    with open(filepath, 'w') as f:
        json.dump(splits, f)
        
def load_splits_from_file(session_id):
    filepath = f'/tmp/splits_{session_id}.json'
    print("LOAD FILEPATH: ", filepath)
    with open(filepath, 'r') as f:
        splits = json.load(f)
    return splits

def delete_splits_file(session_id):
    filepath = f'/tmp/splits_{session_id}.json'
    os.remove(filepath)