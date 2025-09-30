# RAG Test Directory

This directory contains testing and comparison scripts for **RAG (Retrieval-Augmented Generation)** functionality used in the Taiwanese Sign Language Interpretation Service. The scripts focus on semantic search and vector similarity comparison using different approaches.


### Core Scripts

#### `faiss_emb.py`
- **Purpose**: Creates and builds FAISS (Facebook AI Similarity Search) index for semantic search
- **Features**:
  - Loads sentence embeddings from `sign_vectors.pkl`
  - Converts PyTorch tensors to numpy arrays
  - Applies L2 normalization to embeddings
  - Creates cosine similarity index using `IndexFlatIP` (Inner Product)
  - Saves the index as `sign_index_cosine.faiss`
  - Exports metadata (sentences and animation paths) to `sign_metadata.pkl`

#### `faiss_query.py`
- **Purpose**: Performs semantic search queries using the pre-built FAISS index
- **Features**:
  - Uses `gte-Qwen2-1.5B-instruct` model for text embedding
  - Loads FAISS index and metadata
  - Processes user queries and finds top-k similar results
  - Returns cosine similarity scores with corresponding sentences and animation paths
  - Example query: "這桌子少了一隻腳" (This table is missing a leg)

#### `comp.py`
- **Purpose**: Performance comparison between native cosine similarity and FAISS search
- **Features**:
  - **Method A**: Native PyTorch cosine similarity using `sentence_transformers.util.cos_sim`
  - **Method B**: FAISS-based similarity search
  - Measures and compares execution time for both approaches
  - Uses the same embedding model (`gte-Qwen2-1.5B-instruct`)
  - Provides side-by-side results comparison



### Generated Files

#### `sign_index_cosine.faiss`
- Pre-built FAISS index containing normalized embeddings
- Used for fast cosine similarity search
- Created by `faiss_emb.py`

#### Data Dependencies (Referenced but not present)
- `sign_vectors.pkl`: Contains sentences, embeddings, and animation paths
- `sign_metadata.pkl`: Contains sentences and animation paths for FAISS queries

##  Usage Workflow

1. **Build Index**: Run `faiss_emb.py` to create the FAISS index from your embedding data
2. **Query Search**: Use `faiss_query.py` to perform semantic searches
3. **Performance Testing**: Run `comp.py` to compare different search methods ( with or without RAG)


##  Technical Details

### Embedding Model
- **Model**: `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
- **Purpose**: Chinese language understanding and semantic embedding


### Search Method
- **Similarity Metric**: Cosine similarity
- **Normalization**: L2 normalization applied to all vectors
- **Index Type**: FAISS IndexFlatIP 

### Performance Considerations
- FAISS provides significantly faster search for large datasets
- Native PyTorch method suitable for smaller datasets or development testing
- Comparison results help optimize the main RAG system performance

##  Integration

This testing directory supports the main semantic search functionality in the parent `dataset` directory, specifically:
- `semantic_search.py` - Main RAG implementation
- Performance optimization for the deployed service at `http://140.123.105.233:9000`

##  Requirements

- `faiss` (or `faiss-gpu` for CUDA support)
- `sentence-transformers`
- `torch`
- `numpy`
- `pickle`

