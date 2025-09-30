#!/usr/bin/env python

from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os

app = Flask(__name__)

@dataclass
class Doc2VecConfig:
    """Configuration for Doc2Vec model parameters"""
    vector_size: int = 100
    min_count: int = 2
    epochs: int = 10
    alpha: float = 0.025
    min_alpha: float = 0.00025
    window: int = 0
    sample: float = 1e-5
    negative: int = 5

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Doc2VecConfig":
        """Create config from dictionary, filtering out invalid keys"""
        valid_fields = {f.name for f in dataclass.__dict__["__dataclass_fields__"].values() 
                       if f.name in cls.__annotations__}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


def train_doc2vec(documents: Dict[str, List[str]], config: Doc2VecConfig) -> Dict[str, List[float]]:
    """
    Train a Doc2Vec model and return embeddings for all documents.
    
    Args:
        documents: Dictionary mapping document IDs to lists of words
        config: Doc2VecConfig object with model parameters
        
    Returns:
        Dictionary mapping document IDs to their embedding vectors
    """
    # Prepare tagged documents
    tagged_docs = [
        TaggedDocument(words=words, tags=[doc_id])
        for doc_id, words in documents.items()
    ]
    
    # Initialize and train model
    model = Doc2Vec(
		tagged_docs,
        vector_size=config.vector_size,
        epochs=config.epochs,
        alpha=config.alpha,
        min_alpha=config.min_alpha,
        dm=0,
        workers=os.getenv("NUM_WORKERS", 4),
        window=config.window,
        # sample=config.sample,
        negative=config.negative,
    )
    
    # Extract embeddings
    embeddings = {}
    for doc_id in documents.keys():
        embeddings[doc_id] = model.dv[doc_id].tolist()
    
    return embeddings


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/embed", methods=["POST"])
def embed_documents():
    """
    Train Doc2Vec model and return document embeddings.
    
    Expected JSON payload:
    {
        "documents": {
            "doc1": ["word1", "word2", ...],
            "doc2": ["word3", "word4", ...],
            ...
        },
        "config": {
            "vector_size": 100,
            "epochs": 40,
            "alpha": 0.025,
            ...
        }
    }
    
    Returns:
    {
        "embeddings": {
            "doc1": [0.123, -0.456, ...],
            "doc2": [0.789, -0.012, ...],
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or "documents" not in data:
            return jsonify({"error": "Missing documents field in request"}), 400
        
        documents = data["documents"]
        
        if not isinstance(documents, dict):
            return jsonify({"error": "documents must be a dictionary"}), 400
        
        if not documents:
            return jsonify({"error": "No documents provided"}), 400
        
        # Validate document format
        for doc_id, words in documents.items():
            if not isinstance(words, list):
                return jsonify({"error": f"Document {doc_id} must have a list of words"}), 400
            if not all(isinstance(w, str) for w in words):
                return jsonify({"error": f"Document {doc_id} contains non-string words"}), 400
        
        # Get config or use defaults
        config_dict = data.get("config", {})
        config = Doc2VecConfig.from_dict(config_dict)
        
        # Train model and get embeddings
        embeddings = train_doc2vec(documents, config)
        
        return jsonify({"embeddings": embeddings}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/config", methods=["GET"])
def get_default_config():
    """Return default configuration parameters"""
    config = Doc2VecConfig()
    return jsonify({
        "default_config": {
            "vector_size": config.vector_size,
            "min_count": config.min_count,
            "epochs": config.epochs,
            "alpha": config.alpha,
            "min_alpha": config.min_alpha,
            "window": config.window,
            "sample": config.sample,
            "negative": config.negative,
            "hs": config.hs,
        },
        "fixed_params": {
            "dm": config.dm,
            "workers": config.workers
        }
    }), 200


if __name__ == "__main__":
    # For development
    app.run(host="0.0.0.0", port=8080, debug=True)
