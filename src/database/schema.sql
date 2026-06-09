CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS functions (
    id BIGSERIAL PRIMARY KEY,
    library_name TEXT NOT NULL,
    module_name TEXT,
    func_name TEXT NOT NULL,
    full_name TEXT NOT NULL,
    signature TEXT,
    docstring TEXT NOT NULL,
    parameters JSONB DEFAULT '{}'::jsonb,
    returns TEXT,
    examples TEXT,
    source_url TEXT,
    version TEXT,
    chunk_id TEXT UNIQUE,
    search_text TEXT GENERATED ALWAYS AS (
        coalesce(library_name, '') || ' ' ||
        coalesce(module_name, '') || ' ' ||
        coalesce(func_name, '') || ' ' ||
        coalesce(full_name, '') || ' ' ||
        coalesce(signature, '') || ' ' ||
        coalesce(docstring, '') || ' ' ||
        coalesce(examples, '')
    ) STORED,
    search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            coalesce(library_name, '') || ' ' ||
            coalesce(module_name, '') || ' ' ||
            coalesce(func_name, '') || ' ' ||
            coalesce(full_name, '') || ' ' ||
            coalesce(signature, '') || ' ' ||
            coalesce(docstring, '') || ' ' ||
            coalesce(examples, '')
        )
    ) STORED,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS functions_library_name_idx ON functions (library_name);
CREATE INDEX IF NOT EXISTS functions_full_name_idx ON functions (full_name);
CREATE INDEX IF NOT EXISTS functions_search_tsv_idx ON functions USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS functions_embedding_idx
    ON functions USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS benchmark_queries (
    id BIGSERIAL PRIMARY KEY,
    library_name TEXT NOT NULL,
    question TEXT NOT NULL,
    expected_function TEXT NOT NULL,
    expected_answer TEXT,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_documents (
    id BIGSERIAL PRIMARY KEY,
    library_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_url TEXT,
    title TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_examples (
    id BIGSERIAL PRIMARY KEY,
    split TEXT NOT NULL CHECK (split IN ('train', 'test', 'benchmark')),
    task_type TEXT NOT NULL DEFAULT 'retrieval',
    library_name TEXT NOT NULL,
    query TEXT NOT NULL,
    positive_chunk_id TEXT,
    positive_function TEXT NOT NULL,
    answer TEXT,
    hard_negatives JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dataset_examples_split_idx ON dataset_examples (split);
CREATE INDEX IF NOT EXISTS dataset_examples_library_idx ON dataset_examples (library_name);
