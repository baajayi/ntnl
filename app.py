import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Directory containing the data
data_dir = 'data'

# Create a SimpleDirectoryReader instance
reader = SimpleDirectoryReader(data_dir)

# Read data from the directory
documents = reader.load_data()

# Function to chunk texts into smaller pieces
def chunk_text(text, max_tokens=8000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_tokens += len(word) + 1  # +1 for the space
        if current_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to get embeddings using OpenAI API
def get_embeddings(texts):
    all_embeddings = []
    for text in texts:
        chunks = chunk_text(text)
        for chunk in chunks:
            response = client.embeddings.create(
                input=[chunk],
                model='text-embedding-ada-002'
            ).data[0].embedding
            all_embeddings.append(response)
    return all_embeddings

# Check if embeddings.json exists
embeddings_file = 'embeddings.json'
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'r') as f:
        data = json.load(f)
        texts = data['texts']
        embeddings = data['embeddings']
else:
    # Get embeddings for the documents
    texts = [doc.text for doc in documents]
    embeddings = get_embeddings(texts)

    # Save embeddings to file
    with open(embeddings_file, 'w') as f:
        json.dump({'texts': texts, 'embeddings': embeddings}, f)

# Function to generate response using OpenAI GPT-3.5-turbo model with embeddings context
def generate_response(query):
    # Calculate query embedding
    query_embedding = client.embeddings.create(
        input=[query],
        model='text-embedding-ada-002'
    ).data[0].embedding

    # Function to calculate cosine similarity
    def cosine_similarity(vec1, vec2):
        dot_product = sum(p * q for p, q in zip(vec1, vec2))
        magnitude1 = sum(p ** 2 for p in vec1) ** 0.5
        magnitude2 = sum(q ** 2 for q in vec2) ** 0.5
        if not magnitude1 or not magnitude2:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    # Find the most similar embedding
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    most_similar_index = similarities.index(max(similarities))

    # Use the most similar text as context
    context = texts[most_similar_index]

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': '''You are a helpful spiritual assistant.
                                            You are a Lutheran Minister.
                                            You have a context of texts from the Lutheran Church.
                                            Be warm and conversational.
                                            Personalize your response to the user.
                                            Use the context to respond to the query.
                                            If you find no relevant information in the context, say "I do not know the answer to that question."'''},
            {'role': 'system', 'content': context},
            {'role': 'user', 'content': query}
        ]
    )
    return response.choices[0].message.content

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '')
    response = generate_response(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
