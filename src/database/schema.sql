-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop table if exists for clean initialization
DROP TABLE IF EXISTS functions;

-- Create the main table for storing function metadata and embeddings
CREATE TABLE functions (
    id SERIAL PRIMARY KEY,
    library_name VARCHAR(100),
    module_name VARCHAR(255),
    func_name VARCHAR(255),
    signature TEXT,
    docstring TEXT,
    parameters JSONB, 
    search_text TEXT, 
    embedding vector(384) -- 384 dimensions for bge-small-en-v1.5
);

-- Create HNSW index for lightning-fast semantic search
-- cosine distance (vector_cosine_ops) is recommended for text embeddings
CREATE INDEX ON functions USING hnsw (embedding vector_cosine_ops);

-- Create index on library_name for faster metadata filtering
CREATE INDEX ON functions (library_name);
CREATE INDEX idx_fts_search ON functions USING GIN (to_tsvector('english', search_text));