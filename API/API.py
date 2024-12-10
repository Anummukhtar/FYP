import json
from flask import Flask, request, jsonify
import os
import faiss
import numpy as np
from mistralai import Mistral
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


api_key = os.getenv("Mistral_Apikey")
model = "open-mistral-7b"
Mclient = Mistral(api_key=api_key)


documents = [
    {
        "name": "The Time Machine",
        "description": "A man travels through time and witnesses the evolution of humanity.",
        "author": "H.G. Wells",
        "year": 1895,
    },
    {
        "name": "Ender's Game",
        "description": "A young boy is trained to become a military leader in a war against an alien race.",
        "author": "Orson Scott Card",
        "year": 1985,
    },
    
]

encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")


client.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), 
        distance=models.Distance.COSINE,
    ),
)

client.upload_points(
    collection_name="my_books",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)


def retrieve(query, k=1):
    hits = client.query_points(
        collection_name="my_books",
        query=encoder.encode(query).tolist(),
        limit=k,
    ).points

    return [hit.payload for hit in hits]


app = Flask(__name__)


@app.route('/chat', methods=['GET'])
def get_chat():
    input = request.args.get('input')
    if not input:
        return jsonify({"error": "No name provided"}), 400

    retrieved_chat = [doc for doc in documents if doc['name'] == input]
    if retrieved_chat:
        return jsonify(retrieved_chat[0])
    else:
        return jsonify({"error": "chat not found"}), 404


@app.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "No user input provided"}), 400
    
    try:
        
        retrieved_chat = retrieve(user_input, k=3)
        context = " ".join([json.dumps(doc) for doc in retrieved_chat])

        
        messages = [{"role": "system", "content": f"Relevant knowledge: {context}"},
                    {"role": "user", "content": user_input}]

        
        chat_response = Mclient.chat.complete(
            model=model,
            messages=messages
        )

        assistant_response = chat_response.choices[0].message.content
        return jsonify({"response": assistant_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['PUT'])
def create_chat():
    new_chat = request.json
    if not new_chat or not new_chat.get('name'):
        return jsonify({"error": "Invalid chat data"}), 400
    
    
    chatbot.append(new_chat)
    client.upload_points(
        collection_name="my_books",
        points=[
            models.PointStruct(
                id=len(chatbot)-1, 
                vector=encoder.encode(new_chat["description"]).tolist(), 
                payload=new_chat
            )
        ]
    )
    return jsonify({"message": "chat added", "chat": new_chat})


@app.route('/chat', methods=['DELETE'])
def delete_chat():
    name = request.json.get('name')
    if not name:
        return jsonify({"error": "No name provided"}), 400

    
    global chat
    chat = [doc for doc in documents if doc['name'] != name]
    
    
    client.create_collection(
        collection_name="my_books",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    client.upload_points(
        collection_name="my_books",
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )

    return jsonify({"message": f"chat '{name}' deleted."})

app.run(debug=True)
