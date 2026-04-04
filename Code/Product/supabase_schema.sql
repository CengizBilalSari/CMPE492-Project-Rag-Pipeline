-- ============================================================
-- Supabase SQL Schema for Pipeline & Evaluation History
-- Run this in Supabase SQL Editor (Settings → SQL Editor)
-- ============================================================

-- 1. Documents (already created — included here for reference)
CREATE TABLE IF NOT EXISTS documents (
  id            UUID PRIMARY KEY,
  user_id       TEXT NOT NULL,
  filename      TEXT NOT NULL,
  storage_path  TEXT NOT NULL,
  content_type  TEXT NOT NULL DEFAULT 'application/octet-stream',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Pipeline runs — one row per pipeline execution
CREATE TABLE pipeline_runs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       TEXT NOT NULL,
  document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  status        TEXT NOT NULL DEFAULT 'CREATED',
    -- CREATED → CHUNKING → EXTRACTING → RESOLVING → EMBEDDING
    -- → COMMUNITY_DETECTION → SUMMARIZING → COMPLETED | FAILED
  config        JSONB,             -- full PipelineConfig snapshot
  step_times    JSONB,             -- {"1. Chunking": 2.31, ...}
  llm_usage     JSONB,             -- {total_requests, prompt_tokens, completion_tokens, total_tokens}
  neo4j_stats   JSONB,             -- {documents, chunks, entities, relationships, communities}
  error         TEXT,
  started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at  TIMESTAMPTZ
);

-- 3. Evaluation jobs — one row per evaluation run
CREATE TABLE evaluation_jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         TEXT NOT NULL,
  search_types    TEXT[] NOT NULL,          -- e.g. {"global","local","ppr"}
  question_source TEXT NOT NULL,            -- 'auto' | 'custom'
  document_id     UUID REFERENCES documents(id) ON DELETE SET NULL, -- Track source
  status          TEXT NOT NULL DEFAULT 'pending',
    -- pending → generating_questions → evaluating → completed | failed
  progress        TEXT,
  error           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at    TIMESTAMPTZ
);

-- 4. Aggregated metrics per search_type per job (for dashboard charts)
CREATE TABLE evaluation_results (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id            UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
  search_type       TEXT NOT NULL,
  token_cost        INT NOT NULL DEFAULT 0,
  time_per_request  DOUBLE PRECISION NOT NULL DEFAULT 0,
  answer_accuracy   DOUBLE PRECISION NOT NULL DEFAULT 0,
  context_relevance DOUBLE PRECISION NOT NULL DEFAULT 0
);

-- 5. Individual QA pairs (custom-uploaded or auto-generated)
CREATE TABLE qa_pairs (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id               UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
  question             TEXT NOT NULL,
  ground_truth_answer  TEXT NOT NULL,
  source               TEXT NOT NULL DEFAULT 'custom'  -- 'auto' | 'custom'
);

-- 6. Per-question evaluation results (one row per qa_pair × search_type)
CREATE TABLE qa_evaluations (
  id                         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id                     UUID NOT NULL REFERENCES evaluation_jobs(id) ON DELETE CASCADE,
  qa_pair_id                 UUID NOT NULL REFERENCES qa_pairs(id) ON DELETE CASCADE,
  search_type                TEXT NOT NULL,
  rag_answer                 TEXT,
  retrieved_contexts         JSONB,
  answer_correctness_score   DOUBLE PRECISION,
  answer_correctness_reason  TEXT,
  context_relevance_score    DOUBLE PRECISION,
  context_relevance_reason   TEXT,
  latency_ms                 DOUBLE PRECISION NOT NULL DEFAULT 0,
  prompt_tokens              INT NOT NULL DEFAULT 0,
  completion_tokens          INT NOT NULL DEFAULT 0,
  total_tokens               INT NOT NULL DEFAULT 0
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX idx_documents_user          ON documents(user_id);
CREATE INDEX idx_pipeline_runs_user      ON pipeline_runs(user_id);
CREATE INDEX idx_pipeline_runs_doc       ON pipeline_runs(document_id);
CREATE INDEX idx_evaluation_jobs_user    ON evaluation_jobs(user_id);
CREATE INDEX idx_evaluation_results_job  ON evaluation_results(job_id);
CREATE INDEX idx_qa_pairs_job            ON qa_pairs(job_id);
CREATE INDEX idx_qa_evaluations_job      ON qa_evaluations(job_id);
CREATE INDEX idx_qa_evaluations_pair     ON qa_evaluations(qa_pair_id);
