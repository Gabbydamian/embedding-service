from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging
import os
from typing import List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Embedding Service",
    description="A FastAPI service for generating embeddings using sentence-transformers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Request/Response models
class EmbedRequest(BaseModel):
    text: Union[str, List[str]]
    normalize: bool = True

class EmbedResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]
    model_name: str
    text_length: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

@app.on_event("startup")
async def load_model():
    """Load the embedding model on startup"""
    global model
    try:
        logger.info("Loading sentence-transformers model...")
        model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        logger.info(f"Model {model_name} loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embeddings for text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle both single text and batch requests
        if isinstance(request.text, str):
            # Single text
            embedding = model.encode(
                request.text, 
                normalize_embeddings=request.normalize
            ).tolist()
            text_length = len(request.text)
        else:
            # Batch of texts
            embeddings = model.encode(
                request.text, 
                normalize_embeddings=request.normalize
            )
            embedding = embeddings.tolist()
            text_length = sum(len(t) for t in request.text)
        
        return EmbedResponse(
            embedding=embedding,
            model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            text_length=text_length
        )
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.get("/models")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        "max_seq_length": model.max_seq_length,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)