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
        "name": "Mindful Breathing",
        "description": "A guided breathing exercise to help reduce anxiety and stress.",
        "category": "Exercise",
    },
    {
        "name": "Calm Music",
        "description": "A playlist of calming instrumental music to soothe your mind.",
        "category": "Music",
    },
    {
        "name": "Therapist Directory",
        "description": "A list of licensed therapists available for online consultations.",
        "category": "Professional Help",
    },
]


encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")


def initialize_collection():
    client.create_collection(
        collection_name="mental_health_resources",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    client.upload_points(
        collection_name="mental_health_resources",
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )

initialize_collection()


def retrieve_resources(query, k=3):
    hits = client.query_points(
        collection_name="mental_health_resources",
        query=encoder.encode(query).tolist(),
        limit=k,
    ).points
    return [hit.payload for hit in hits]


app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chatbot():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "No user input provided"}), 400

    try:
        
        retrieved_resources = retrieve_resources(user_input, k=3)
        context = "\n".join([f"{doc['name']}: {doc['description']}" for doc in retrieved_resources])

        
        messages = [
            {"role": "system", "content": "You are a mental health assistant providing emotional support and personalized recommendations."},
            {"role": "system", "content": f"Relevant resources:\n{context}"},
            {"role": "user", "content": user_input}
        ]

        
        chat_response = Mclient.chat.complete(
            model=model,
            messages=messages
        )

        assistant_response = chat_response.choices[0].message.content
        return jsonify({"response": assistant_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/resources', methods=['POST'])
def add_resource():
    new_resource = request.json
    if not new_resource or not new_resource.get('name') or not new_resource.get('description'):
        return jsonify({"error": "Invalid resource data"}), 400

    documents.append(new_resource)
    client.upload_points(
        collection_name="mental_health_resources",
        points=[
            models.PointStruct(
                id=len(documents)-1,
                vector=encoder.encode(new_resource["description"]).tolist(),
                payload=new_resource
            )
        ]
    )
    return jsonify({"message": "Resource added", "resource": new_resource})

@app.route('/resources', methods=['DELETE'])
def delete_resource():
    name = request.json.get('name')
    if not name:
        return jsonify({"error": "No name provided"}), 400

    global documents
    documents = [doc for doc in documents if doc['name'] != name]

    initialize_collection() 
    return jsonify({"message": f"Resource '{name}' deleted."})

if __name__ == '__main__':
    app.run(debug=True)
