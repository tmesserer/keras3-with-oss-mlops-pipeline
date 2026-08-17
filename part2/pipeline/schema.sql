-- Part 2 tables. Created idempotently at service startup, so there is no migration step
-- and no ordering dependency between the services that use them.
--
-- Label Studio manages its own tables in this same database; everything here is prefixed
-- to stay out of its way.

-- Every transaction the endpoint scored. This is the spine: the review queue selects from
-- it, the app's dashboard aggregates it, and the retraining snapshot joins it to verdicts.
CREATE TABLE IF NOT EXISTS scored_transactions (
    id              BIGSERIAL PRIMARY KEY,
    trans_num       TEXT UNIQUE,            -- natural key from the dataset, when present
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    probability     REAL        NOT NULL,
    decision        TEXT        NOT NULL CHECK (decision IN ('approve','review','decline')),
    threshold_review  REAL      NOT NULL,
    threshold_decline REAL      NOT NULL,
    latency_ms      REAL,
    -- The raw request, kept whole: the retraining snapshot needs the original features and
    -- storing them as JSONB avoids a schema migration every time a feature is added.
    payload         JSONB       NOT NULL,
    -- Ground truth when the caller happens to know it (replaying a labelled test set).
    -- NULL in any real deployment -- that absence is what the review queue exists to fill.
    label           SMALLINT
);

CREATE INDEX IF NOT EXISTS scored_decision_idx ON scored_transactions (decision, scored_at DESC);
CREATE INDEX IF NOT EXISTS scored_at_idx       ON scored_transactions (scored_at DESC);

-- Analyst verdicts coming back out of Label Studio. One row per completed annotation.
CREATE TABLE IF NOT EXISTS verdicts (
    id              BIGSERIAL PRIMARY KEY,
    scored_id       BIGINT      NOT NULL REFERENCES scored_transactions (id) ON DELETE CASCADE,
    verdict         SMALLINT    NOT NULL CHECK (verdict IN (0, 1)),   -- 1 = confirmed fraud
    decided_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    annotator       TEXT,
    ls_task_id      BIGINT,                 -- Label Studio task, for reconciliation
    -- Which model version flagged it. Lets a later run ask "what did v3 get wrong?"
    flagged_by      TEXT,
    UNIQUE (scored_id)
);

CREATE INDEX IF NOT EXISTS verdicts_decided_idx ON verdicts (decided_at DESC);

-- What the consuming application saw. Deliberately separate from scored_transactions:
-- that table is what the SERVER did, this is what the CLIENT experienced. The gap between
-- server_ms and client_ms is the network and queueing the model never sees, and rows where
-- outcome != 'scored' have no server-side counterpart at all -- those are exactly the
-- transactions the app had to decide without a model.
CREATE TABLE IF NOT EXISTS app_requests (
    id            BIGSERIAL PRIMARY KEY,
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    trans_num     TEXT,
    outcome       TEXT NOT NULL CHECK (outcome IN ('scored','fallback','shed')),
    decision      TEXT NOT NULL,
    client_ms     REAL,
    server_ms     REAL,
    model_version TEXT,
    breaker_state TEXT,
    error         TEXT
);

CREATE INDEX IF NOT EXISTS app_requests_at_idx      ON app_requests (requested_at DESC);
CREATE INDEX IF NOT EXISTS app_requests_outcome_idx ON app_requests (outcome, requested_at DESC);
