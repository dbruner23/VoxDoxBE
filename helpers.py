import json
from langchain_core.messages import HumanMessage

def prepare_whole_document_for_frontend(documents):
    return [
        {
            "pageContent": doc.page_content,
            "source": doc.metadata['source'],
            "pageNumber": doc.metadata['page']
        } 
        for doc in documents
    ]
    
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
        return {"content": message["answer"], "role": "ai"}