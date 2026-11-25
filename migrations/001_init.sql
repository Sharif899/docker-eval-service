-- 001_init.sql — initial schema for docker-eval-service.
--
-- evaluations: one row per scored response, with the aggregate
--   outcome (mean, total, pass). Per-criterion scores live in
--   `scores` so they stay queryable.
-- scores: one row per criterion per evaluation.
-- comparisons: one row per pairwise comparison, linking the two
--   evaluation rows it was computed from. Per-criterion
--   comparisons are stored as JSONB (they are only ever read
--   back whole, never filtered on).
-- schema_migrations: applied-migration tracking, so startup
--   applies each file exactly once.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    reference TEXT,
    mean_score NUMERIC(3, 2) NOT NULL,
    total_score INTEGER NOT NULL,
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations (model);
CREATE INDEX IF NOT EXISTS idx_evaluations_task_type ON evaluations (task_type);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations (created_at DESC);

CREATE TABLE IF NOT EXISTS scores (
    id UUID PRIMARY KEY,
    evaluation_id UUID NOT NULL REFERENCES evaluations (id) ON DELETE CASCADE,
    criterion TEXT NOT NULL,
    score INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
    rationale TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_scores_evaluation_id ON scores (evaluation_id);

CREATE TABLE IF NOT EXISTS comparisons (
    id UUID PRIMARY KEY,
    evaluation_a_id UUID NOT NULL REFERENCES evaluations (id) ON DELETE CASCADE,
    evaluation_b_id UUID NOT NULL REFERENCES evaluations (id) ON DELETE CASCADE,
    model_a TEXT NOT NULL,
    model_b TEXT NOT NULL,
    task_type TEXT NOT NULL,
    prompt TEXT NOT NULL,
    winner TEXT NOT NULL CHECK (winner IN ('A', 'B', 'tie')),
    mean_a NUMERIC(3, 2) NOT NULL,
    mean_b NUMERIC(3, 2) NOT NULL,
    criteria JSONB NOT NULL,
    rationale TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_comparisons_created_at ON comparisons (created_at DESC);
