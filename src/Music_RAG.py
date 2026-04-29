from recommender import load_songs, recommend_songs
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
from typing import List
import os
import chromadb

load_dotenv("Music_Key.env")

#Start by loading songs and converting features into text

def list_to_text_conv(csv_path: str) -> List[str]:
    songs = load_songs(csv_path)
    descriptions = []
    for song in songs:
        text = (
            f"Title: {song['title']}. "
            f"Artist: {song['artist']}. "
            f"Genre: {song['genre']}. "
            f"Mood: {song['mood']}. "
            f"Energy: {song['energy']}."
        )
        descriptions.append(text)
    return descriptions


# Embed descriptions into vectors

def embed_songs(descriptions: List[str]) -> List[List[float]]:
    ef = DefaultEmbeddingFunction()
    return [[float(v) for v in vec] for vec in ef(descriptions)]


# Store embeds in a vector store/database

def store_vector(vectors: List[List[float]], descriptions: List[str]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="songs")
    collection.add(
        embeddings=vectors,
        documents=descriptions,
        ids=[str(i) for i in range(len(vectors))]
    )
    return collection


# Given user profile, embed it and retrieve nearest songs

def embed_prompt(prompt: str) -> List[float]:
    ef = DefaultEmbeddingFunction()
    return [float(v) for v in ef([prompt])[0]]


def nearest_songs(query_vector: List[float], collection: chromadb.Collection, k: int = 5) -> List[str]:
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents"]
    )
    return results["documents"][0]


# Pass retrieved songs as context to an LLM to generate final recs

def format_retrieved(results: List[str]) -> List[str]:
    formatted = []
    for description in results:
        parts = {
            segment.split(": ")[0].strip(): segment.split(": ")[1].strip(" .")
            for segment in description.split(". ")
            if ": " in segment
        }
        line = (
            f"Title: {parts.get('Title', 'N/A')} | "
            f"Genre: {parts.get('Genre', 'N/A')} | "
            f"Mood: {parts.get('Mood', 'N/A')}"
        )
        formatted.append(line)
    return formatted


def generate_recs(prompt: str, formatted_songs: List[str]) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    context = "\n".join(f"{i+1}. {song}" for i, song in enumerate(formatted_songs))
    user_message = (
        f"User request: {prompt}\n\n"
        f"Candidate songs:\n{context}\n\n"
        "Return the top 5 recommendations ranked from best to worst. "
        "For each song, explain why it matches the user's request."
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a music recommendation assistant. "
                    "You will be given a list of candidate songs and a user's request. "
                    "Rank the top 5 songs that best match the request and explain why each one fits."
                )
            },
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content


def main(prompt: str) -> None:
    descriptions = list_to_text_conv("songs.csv")
    vectors = embed_songs(descriptions)
    collection = store_vector(vectors, descriptions)
    query_vector = embed_prompt(prompt)
    results = nearest_songs(query_vector, collection)
    formatted = format_retrieved(results)
    output = generate_recs(prompt, formatted)
    print(output)
