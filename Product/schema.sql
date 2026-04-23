-- ============================================================
-- PostgreSQL schema for GraphRAG pipeline & evaluation history
-- Mounted at /docker-entrypoint-initdb.d/schema.sql so it runs
-- automatically on first container start.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Chats (local workspaces) ──────────────────────────────────
-- A single local user can create many named chat bases.
CREATE TABLE IF NOT EXISTS chats (
  chat_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name            TEXT NOT NULL,
  embedding_model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name)
);


-- ── Documents (metadata; actual bytes live on the local FS) ─
CREATE TABLE IF NOT EXISTS documents (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chat_id      UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  path         TEXT NOT NULL,
  content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_chat ON documents(chat_id);

-- ── Pipeline runs ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chat_id      UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
  document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  status       TEXT NOT NULL DEFAULT 'CREATED',
  config       JSONB,
  step_times   JSONB,
  llm_usage    JSONB,
  neo4j_stats  JSONB,
  error        TEXT,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_chat ON pipeline_runs(chat_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_doc  ON pipeline_runs(document_id);

-- ── Evaluation jobs ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evaluation_jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chat_id         UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
  search_types    TEXT[] NOT NULL,
  question_source TEXT NOT NULL,
  document_id     UUID REFERENCES documents(id) ON DELETE SET NULL,
  status          TEXT NOT NULL DEFAULT 'pending',
  progress        TEXT,
  error           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_evaluation_jobs_chat ON evaluation_jobs(chat_id);

-- ── Aggregated results per search_type per job ──────────────
CREATE TABLE IF NOT EXISTS evaluation_results (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id            UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
  search_type       TEXT NOT NULL,
  token_cost        INT NOT NULL DEFAULT 0,
  time_per_request  DOUBLE PRECISION NOT NULL DEFAULT 0,
  answer_accuracy   DOUBLE PRECISION NOT NULL DEFAULT 0,
  context_relevance DOUBLE PRECISION NOT NULL DEFAULT 0,
  prompt_tokens     INT NOT NULL DEFAULT 0,
  completion_tokens INT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_evaluation_results_job ON evaluation_results(job_id);

-- ── Individual QA pairs ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS qa_pairs (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id              UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
  question            TEXT NOT NULL,
  ground_truth_answer TEXT NOT NULL,
  source              TEXT NOT NULL DEFAULT 'custom'
);

CREATE INDEX IF NOT EXISTS idx_qa_pairs_job ON qa_pairs(job_id);

-- ── Per-question evaluation results ─────────────────────────
CREATE TABLE IF NOT EXISTS qa_evaluations (
  id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id                    UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
  qa_pair_id                UUID NOT NULL REFERENCES qa_pairs(id) ON DELETE CASCADE,
  search_type               TEXT NOT NULL,
  rag_answer                TEXT,
  rag_reasoning             TEXT,
  retrieved_contexts        JSONB,
  answer_correctness_score  DOUBLE PRECISION,
  answer_correctness_reason TEXT,
  context_relevance_score   DOUBLE PRECISION,
  context_relevance_reason  TEXT,
  latency_ms                DOUBLE PRECISION NOT NULL DEFAULT 0,
  prompt_tokens             INT NOT NULL DEFAULT 0,
  completion_tokens         INT NOT NULL DEFAULT 0,
  total_tokens              INT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_qa_evaluations_job  ON qa_evaluations(job_id);
CREATE INDEX IF NOT EXISTS idx_qa_evaluations_pair ON qa_evaluations(qa_pair_id);
