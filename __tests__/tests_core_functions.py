import os
import pytest
from flask import Flask, jsonify
from app import query
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
index_name = os.getenv("PINECONE_INDEX_NAME")

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    return app

@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })

    # other setup

    yield app

@pytest.fixture
def client(app):
    return app.test_client()

def test_query_with_documents(client):
    # Set up the necessary dependencies
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    # Load your documents into the vectorstore

    with app.test_request_context():
        # Create a new session for this test
        app.preprocess_request()

        # Call the query function with your desired input
        question = "What does it mean to park a church?"
        response_data, status_code = query()

        # Assert the expected output
        assert status_code == 200
        assert response_data == "Expected response"