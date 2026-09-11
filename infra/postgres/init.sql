-- Runs once, on first container start (docker-entrypoint-initdb.d convention).
-- Enables pgvector for the RAG module (§7.9) in the same Postgres instance the
-- Core API uses — one database, `rag` schema, per README §7.9 / §5 "one writer
-- per store" (the RAG module is the only thing that writes to the rag schema).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag;

-- pqc-core-api's own tables are created by its SQLx migrations
-- (rust/crates/pqc-core-api/migrations/), not here.
