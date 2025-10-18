# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "numpy",
# ]
# ///

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import httpx
import json

app = FastAPI(title="TypeScript Book RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for chunks and embeddings
chunks_store = []
embeddings_store = []

# aipipe.org configuration
AIPIPE_URL = "https://aipipe.org/openai/v1/embeddings"
AIPIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDI1MThAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.X1MMvFrsIHOpwGS4qtYIddDlXwIHq1Nl2KrDMHax0Fs"

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts using aipipe.org"""
    headers = {
        "Authorization": AIPIPE_TOKEN,
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "text-embedding-3-small",
        "input": texts,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(AIPIPE_URL, headers=headers, json=payload)
        data = response.json()
    
    if "data" not in data:
        raise Exception(f"Error from aipipe.org: {data}")
    
    return [item["embedding"] for item in data["data"]]

@app.on_event("startup")
async def load_chunks():
    """Load chunks from chunks.json and generate embeddings"""
    global chunks_store, embeddings_store
    
    try:
        with open("chunks.json", "r") as f:
            chunks_store = [json.loads(line) for line in f.readlines()]
        
        print(f"Loaded {len(chunks_store)} chunks from chunks.json")
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 50
        print("Generating embeddings...")
        
        for i in range(0, len(chunks_store), batch_size):
            batch = chunks_store[i:i+batch_size]
            texts = [chunk["content"] for chunk in batch]
            
            try:
                batch_embeddings = await get_embeddings(texts)
                embeddings_store.extend(batch_embeddings)
                print(f"Processed {min(i+batch_size, len(chunks_store))}/{len(chunks_store)} chunks")
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # If batch fails, try one by one
                for chunk in batch:
                    try:
                        emb = await get_embeddings([chunk["content"]])
                        embeddings_store.append(emb[0])
                    except:
                        # Use zero vector as fallback
                        embeddings_store.append([0.0] * 1536)
        
        print(f"Successfully generated {len(embeddings_store)} embeddings!")
        
    except FileNotFoundError:
        print("Warning: chunks.json not found. Using empty store.")
        chunks_store = []
        embeddings_store = []

@app.get("/search")
async def search(q: str = Query(..., description="The question to search for")):
    """
    Search endpoint that returns relevant documentation excerpts
    
    Example: /search?q=What does the author affectionately call the => syntax?
    """
    if not chunks_store:
        return {
            "answer": "No documentation loaded. Please ensure chunks.json exists in the same directory.",
            "sources": None
        }
    
    try:
        # Get embedding for the query
        query_embeddings = await get_embeddings([q])
        query_embedding = query_embeddings[0]
        
        # Calculate similarities with all chunks
        similarities = []
        for i, chunk_embedding in enumerate(embeddings_store):
            sim = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 most similar chunks
        top_chunks = similarities[:3]
        
        # Combine the top chunks into answer
        answer_parts = []
        sources = []
        
        for idx, score in top_chunks:
            chunk = chunks_store[idx]
            answer_parts.append(chunk["content"])
            sources.append({
                "id": chunk["id"],
                "similarity": round(score, 4)
            })
        
        # Join chunks with clear separators
        answer = "\n\n---\n\n".join(answer_parts)
        
        return {
            "answer": answer,
            "sources": sources,
            "query": q
        }
    
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": None,
            "query": q
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "chunks_loaded": len(chunks_store),
        "embeddings_generated": len(embeddings_store),
        "message": "RAG API for TypeScript Book using aipipe.org",
        "usage": "GET /search?q=your question here"
    }
