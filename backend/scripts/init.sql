-- ============================================================
-- Схема БД чат-бота службы поддержки интернет-магазина
-- PostgreSQL 15 + pgvector
-- Автор: Зорин С.И.
-- Версия: 1.0
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- 0. РАСШИРЕНИЯ И ТИПЫ
-- ────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;         -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;        -- триграммный поиск (для FAQ matching)

-- Перечислимые типы
CREATE TYPE message_direction AS ENUM ('in', 'out');

CREATE TYPE message_source AS ENUM (
    'user',          -- от клиента
    'bot_faq',       -- ответ бота из FAQ
    'bot_rag',       -- ответ бота из RAG
    'bot_api',       -- ответ бота из API (заказы)
    'bot_scenario',  -- ответ бота из сценария (кнопки, приветствие)
    'operator'       -- от оператора
);

CREATE TYPE channel_type AS ENUM (
    'web_widget',
    'telegram',
    'api'
);

CREATE TYPE escalation_reason AS ENUM (
    'user_request',       -- пользователь попросил оператора
    'low_confidence',     -- низкая уверенность классификатора
    'max_failures',       -- превышен лимит непонятых сообщений
    'technical_issue',    -- категория = тех. проблема
    'negative_sentiment', -- негативная реакция пользователя
    'max_order_attempts', -- превышен лимит попыток ввода номера заказа
    'rag_no_answer'       -- RAG не нашёл ответа
);

CREATE TYPE escalation_trigger AS ENUM (
    'automatic',   -- система сама инициировала
    'manual'       -- пользователь запросил
);

CREATE TYPE feedback_rating AS ENUM (
    'positive',    -- 👍 / 1
    'negative'     -- 👎 / -1
);

CREATE TYPE feedback_reason AS ENUM (
    'inaccurate',       -- неточный ответ
    'irrelevant',       -- не по теме
    'incomplete',       -- неполный ответ
    'too_slow',         -- долго ждал
    'other'             -- другое
);

CREATE TYPE config_value_type AS ENUM (
    'string',
    'integer',
    'float',
    'boolean',
    'json'
);

CREATE TYPE generator_mode AS ENUM (
    'template',
    'local_llm',
    'external_llm'
);


-- ────────────────────────────────────────────────────────────
-- 1. СПРАВОЧНИКИ
-- ────────────────────────────────────────────────────────────

-- Категории обращений (CAT-01..CAT-10)
CREATE TABLE categories (
    id          SERIAL       PRIMARY KEY,
    code        VARCHAR(50)  NOT NULL UNIQUE,
    label_ru    VARCHAR(100) NOT NULL,
    label_en    VARCHAR(100) NOT NULL,
    description TEXT,
    sort_order  SMALLINT     NOT NULL DEFAULT 0,
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ
);

COMMENT ON TABLE  categories IS 'Справочник категорий обращений для ML-классификатора (Level 1)';
COMMENT ON COLUMN categories.code IS 'Уникальный код категории (например, order_status, payment_refund)';
COMMENT ON COLUMN categories.sort_order IS 'Порядок отображения в кнопках уточнения';

-- Статусы заказов
CREATE TABLE order_statuses (
    id          SERIAL      PRIMARY KEY,
    code        VARCHAR(30) NOT NULL UNIQUE,
    label_ru    VARCHAR(50) NOT NULL,
    label_en    VARCHAR(50) NOT NULL,
    sort_order  SMALLINT    NOT NULL DEFAULT 0,
    is_terminal BOOLEAN     NOT NULL DEFAULT FALSE
);

COMMENT ON TABLE  order_statuses IS 'Справочник статусов заказа';
COMMENT ON COLUMN order_statuses.is_terminal IS 'TRUE для финальных статусов (delivered, cancelled, returned)';


-- ────────────────────────────────────────────────────────────
-- 2. FAQ
-- ────────────────────────────────────────────────────────────

CREATE TABLE faq_entries (
    id              SERIAL       PRIMARY KEY,
    category_id     INTEGER      REFERENCES categories(id)
                                 ON DELETE SET NULL,
    trigger_phrases JSONB        NOT NULL,
    answer_text     TEXT         NOT NULL,
    buttons         JSONB,
    priority        SMALLINT     NOT NULL DEFAULT 0,
    hit_count       INTEGER      NOT NULL DEFAULT 0,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ,

    -- Валидация: trigger_phrases должен быть массивом строк
    CONSTRAINT chk_trigger_phrases_is_array
        CHECK (jsonb_typeof(trigger_phrases) = 'array'
               AND jsonb_array_length(trigger_phrases) > 0),

    -- Валидация: buttons — массив объектов или NULL
    CONSTRAINT chk_buttons_format
        CHECK (buttons IS NULL
               OR jsonb_typeof(buttons) = 'array')
);

COMMENT ON TABLE  faq_entries IS 'FAQ-записи для fast-path обработки (до ML/RAG)';
COMMENT ON COLUMN faq_entries.trigger_phrases IS 'JSON-массив триггерных фраз: ["как вернуть товар", "возврат товара", "хочу вернуть"]';
COMMENT ON COLUMN faq_entries.buttons IS 'JSON-массив кнопок: [{"text": "Да", "value": "yes"}, ...]';
COMMENT ON COLUMN faq_entries.hit_count IS 'Счётчик срабатываний — для аналитики популярности';
COMMENT ON COLUMN faq_entries.priority IS 'Приоритет при множественном совпадении (выше = важнее)';

-- Индекс для GIN-поиска по триггерным фразам
CREATE INDEX idx_faq_trigger_gin ON faq_entries USING GIN (trigger_phrases jsonb_path_ops);
CREATE INDEX idx_faq_active ON faq_entries (is_active) WHERE is_active = TRUE;


-- ────────────────────────────────────────────────────────────
-- 3. БАЗА ЗНАНИЙ (RAG)
-- ────────────────────────────────────────────────────────────

CREATE TABLE kb_documents (
    id           SERIAL       PRIMARY KEY,
    category_id  INTEGER      REFERENCES categories(id)
                              ON DELETE SET NULL,
    title        VARCHAR(255) NOT NULL,
    source_file  VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64)  NOT NULL,
    chunk_count  INTEGER      NOT NULL DEFAULT 0,
    is_active    BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ
);

COMMENT ON TABLE  kb_documents IS 'Документы базы знаний (политики, инструкции, описания)';
COMMENT ON COLUMN kb_documents.content_hash IS 'SHA-256 хеш содержимого — для обнаружения изменений при переиндексации';
COMMENT ON COLUMN kb_documents.chunk_count IS 'Количество чанков после разбиения (денормализация для быстрого отображения)';

CREATE TABLE kb_chunks (
    id           SERIAL       PRIMARY KEY,
    document_id  INTEGER      NOT NULL
                              REFERENCES kb_documents(id)
                              ON DELETE CASCADE,
    chunk_index  SMALLINT     NOT NULL,
    content      TEXT         NOT NULL,
    embedding    VECTOR(384)  NOT NULL,
    token_count  SMALLINT     NOT NULL,
    char_count   INTEGER      NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Уникальность: один документ — один индекс чанка
    CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index),

    -- Валидация
    CONSTRAINT chk_chunk_index_positive CHECK (chunk_index >= 0),
    CONSTRAINT chk_token_count_positive CHECK (token_count > 0),
    CONSTRAINT chk_char_count_positive  CHECK (char_count > 0)
);

COMMENT ON TABLE  kb_chunks IS 'Чанки (фрагменты) документов с эмбеддингами для векторного поиска';
COMMENT ON COLUMN kb_chunks.embedding IS 'Вектор эмбеддинга (384 dim — paraphrase-multilingual-MiniLM-L12-v2)';

-- IVFFlat индекс для приближённого поиска ближайших соседей
-- lists = sqrt(N) оптимально; для прототипа ≤ 10000 чанков → lists = 100
CREATE INDEX idx_kb_chunks_embedding
    ON kb_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

CREATE INDEX idx_kb_chunks_document ON kb_chunks (document_id);


-- ────────────────────────────────────────────────────────────
-- 4. ЗАКАЗЫ
-- ────────────────────────────────────────────────────────────

CREATE TABLE customers (
    id           SERIAL       PRIMARY KEY,
    external_id  VARCHAR(50)  NOT NULL UNIQUE,
    email        VARCHAR(255),
    phone        VARCHAR(20),
    first_name   VARCHAR(100),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  customers IS 'Клиенты интернет-магазина (минимальный профиль)';
COMMENT ON COLUMN customers.external_id IS 'ID клиента из внешней системы (CRM / e-commerce)';
COMMENT ON COLUMN customers.email IS '⚠️ PII — маскировать в логах';
COMMENT ON COLUMN customers.phone IS '⚠️ PII — маскировать в логах';

CREATE TABLE orders (
    id                  SERIAL        PRIMARY KEY,
    order_number        VARCHAR(20)   NOT NULL UNIQUE,
    customer_id         INTEGER       NOT NULL
                                      REFERENCES customers(id)
                                      ON DELETE RESTRICT,
    status_id           INTEGER       NOT NULL
                                      REFERENCES order_statuses(id)
                                      ON DELETE RESTRICT,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    estimated_delivery  DATE,
    delivered_at        TIMESTAMPTZ,
    tracking_number     VARCHAR(50),
    items_count         SMALLINT      NOT NULL DEFAULT 0,
    total_amount        NUMERIC(12,2) NOT NULL DEFAULT 0.00,
    currency            VARCHAR(3)    NOT NULL DEFAULT 'RUB',
    delivery_address    TEXT,
    updated_at          TIMESTAMPTZ,

    -- Валидации
    CONSTRAINT chk_order_number_format
        CHECK (order_number ~ '^\d{4,8}$'),
    CONSTRAINT chk_total_positive
        CHECK (total_amount >= 0),
    CONSTRAINT chk_items_positive
        CHECK (items_count >= 0)
);

COMMENT ON TABLE  orders IS 'Заказы интернет-магазина (для отслеживания статуса через бота)';
COMMENT ON COLUMN orders.order_number IS 'Номер заказа, видимый клиенту (например, 123456)';
COMMENT ON COLUMN orders.delivery_address IS '⚠️ PII — маскировать в логах';

CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_status   ON orders (status_id);
CREATE INDEX idx_orders_number   ON orders (order_number);

CREATE TABLE order_items (
    id           SERIAL        PRIMARY KEY,
    order_id     INTEGER       NOT NULL
                               REFERENCES orders(id)
                               ON DELETE CASCADE,
    product_name VARCHAR(255)  NOT NULL,
    quantity     SMALLINT      NOT NULL DEFAULT 1,
    unit_price   NUMERIC(10,2) NOT NULL,

    CONSTRAINT chk_quantity_positive CHECK (quantity > 0),
    CONSTRAINT chk_price_positive    CHECK (unit_price >= 0)
);

COMMENT ON TABLE order_items IS 'Позиции заказа (состав)';

CREATE INDEX idx_order_items_order ON order_items (order_id);

CREATE TABLE order_status_history (
    id         SERIAL      PRIMARY KEY,
    order_id   INTEGER     NOT NULL
                           REFERENCES orders(id)
                           ON DELETE CASCADE,
    status_id  INTEGER     NOT NULL
                           REFERENCES order_statuses(id)
                           ON DELETE RESTRICT,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    comment    TEXT
);

COMMENT ON TABLE order_status_history IS 'История изменений статуса заказа (для отображения таймлайна)';

CREATE INDEX idx_osh_order     ON order_status_history (order_id);
CREATE INDEX idx_osh_changed   ON order_status_history (changed_at);


-- ────────────────────────────────────────────────────────────
-- 5. ДИАЛОГИ
-- ────────────────────────────────────────────────────────────

CREATE TABLE sessions (
    id                  SERIAL          PRIMARY KEY,
    session_ext_id      VARCHAR(100)    NOT NULL UNIQUE,
    customer_id         INTEGER         REFERENCES customers(id)
                                        ON DELETE SET NULL,
    channel             channel_type    NOT NULL DEFAULT 'web_widget',
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    message_count       INTEGER         NOT NULL DEFAULT 0,
    is_escalated        BOOLEAN         NOT NULL DEFAULT FALSE,
    is_resolved         BOOLEAN         NOT NULL DEFAULT FALSE,
    final_category      VARCHAR(50),
    satisfaction_rating SMALLINT,

    -- Валидации
    CONSTRAINT chk_rating_range
        CHECK (satisfaction_rating IS NULL
               OR (satisfaction_rating >= 1
                   AND satisfaction_rating <= 5)),
    CONSTRAINT chk_end_after_start
        CHECK (ended_at IS NULL
               OR ended_at >= started_at)
);

COMMENT ON TABLE  sessions IS 'Сессии диалогов (одна сессия = один разговор клиента с ботом)';
COMMENT ON COLUMN sessions.session_ext_id IS 'Внешний ID сессии из JAICP';
COMMENT ON COLUMN sessions.final_category IS 'Итоговая категория обращения (определена по ходу диалога)';

CREATE INDEX idx_sessions_customer ON sessions (customer_id);
CREATE INDEX idx_sessions_started  ON sessions (started_at);
CREATE INDEX idx_sessions_escalated ON sessions (is_escalated) WHERE is_escalated = TRUE;

CREATE TABLE messages (
    id               SERIAL            PRIMARY KEY,
    session_id       INTEGER           NOT NULL
                                       REFERENCES sessions(id)
                                       ON DELETE CASCADE,
    direction        message_direction NOT NULL,
    content          TEXT              NOT NULL,
    source           message_source    NOT NULL,
    detected_intent  VARCHAR(50),
    confidence       REAL,
    response_time_ms INTEGER,
    created_at       TIMESTAMPTZ       NOT NULL DEFAULT NOW(),

    -- Валидации
    CONSTRAINT chk_confidence_range
        CHECK (confidence IS NULL
               OR (confidence >= 0.0
                   AND confidence <= 1.0)),
    CONSTRAINT chk_response_time_positive
        CHECK (response_time_ms IS NULL
               OR response_time_ms >= 0)
);

COMMENT ON TABLE  messages IS 'Сообщения в диалоговых сессиях';
COMMENT ON COLUMN messages.direction IS 'in = от клиента, out = от бота/оператора';
COMMENT ON COLUMN messages.source IS 'Источник ответа: faq, rag, api, scenario, operator';
COMMENT ON COLUMN messages.detected_intent IS 'Распознанный интент (мета или категория)';
COMMENT ON COLUMN messages.content IS '⚠️ Может содержать PII — маскировать при экспорте';

CREATE INDEX idx_messages_session   ON messages (session_id);
CREATE INDEX idx_messages_created   ON messages (created_at);
CREATE INDEX idx_messages_intent    ON messages (detected_intent)
    WHERE detected_intent IS NOT NULL;


-- ────────────────────────────────────────────────────────────
-- 6. ML-ЛОГИ
-- ────────────────────────────────────────────────────────────

CREATE TABLE classification_logs (
    id                  SERIAL       PRIMARY KEY,
    message_id          INTEGER      UNIQUE
                                     REFERENCES messages(id)
                                     ON DELETE CASCADE,
    input_text          TEXT         NOT NULL,
    predicted_category  VARCHAR(50)  NOT NULL,
    confidence          REAL         NOT NULL,
    top_3_categories    JSONB        NOT NULL,
    extracted_entities  JSONB,
    processing_time_ms  INTEGER      NOT NULL,
    model_version       VARCHAR(20)  NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_cls_confidence_range
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT chk_cls_proc_time
        CHECK (processing_time_ms >= 0)
);

COMMENT ON TABLE  classification_logs IS 'Лог результатов ML-классификации (для анализа качества модели)';
COMMENT ON COLUMN classification_logs.top_3_categories IS '[{"category": "order_status", "confidence": 0.92}, ...]';
COMMENT ON COLUMN classification_logs.extracted_entities IS '{"order_number": "123456", "email": "***@***.ru"}';
COMMENT ON COLUMN classification_logs.model_version IS 'Версия модели (например, v1.0, v1.1) — для A/B тестов';

CREATE INDEX idx_cls_logs_category   ON classification_logs (predicted_category);
CREATE INDEX idx_cls_logs_created    ON classification_logs (created_at);
CREATE INDEX idx_cls_logs_confidence ON classification_logs (confidence);
CREATE INDEX idx_cls_logs_model      ON classification_logs (model_version);

CREATE TABLE rag_query_logs (
    id                 SERIAL         PRIMARY KEY,
    message_id         INTEGER        UNIQUE
                                      REFERENCES messages(id)
                                      ON DELETE CASCADE,
    query_text         TEXT           NOT NULL,
    top_k_requested    SMALLINT       NOT NULL DEFAULT 5,
    retrieved_chunk_ids JSONB,
    similarity_scores  JSONB,
    max_similarity     REAL           NOT NULL,
    generated_answer   TEXT,
    is_confident       BOOLEAN        NOT NULL,
    generator_mode     generator_mode NOT NULL DEFAULT 'template',
    processing_time_ms INTEGER        NOT NULL,
    created_at         TIMESTAMPTZ    NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rag_max_sim_range
        CHECK (max_similarity >= 0.0 AND max_similarity <= 1.0),
    CONSTRAINT chk_rag_proc_time
        CHECK (processing_time_ms >= 0)
);

COMMENT ON TABLE  rag_query_logs IS 'Лог RAG-запросов (для оценки качества retrieval и generation)';
COMMENT ON COLUMN rag_query_logs.retrieved_chunk_ids IS '[42, 43, 78] — ID найденных чанков';
COMMENT ON COLUMN rag_query_logs.similarity_scores IS '[0.87, 0.82, 0.71] — cosine similarity';
COMMENT ON COLUMN rag_query_logs.is_confident IS 'TRUE если max_similarity >= порога (0.70)';


-- ────────────────────────────────────────────────────────────
-- 7. ОБРАТНАЯ СВЯЗЬ
-- ────────────────────────────────────────────────────────────

CREATE TABLE feedback (
    id            SERIAL           PRIMARY KEY,
    session_id    INTEGER          NOT NULL
                                   REFERENCES sessions(id)
                                   ON DELETE CASCADE,
    message_id    INTEGER          REFERENCES messages(id)
                                   ON DELETE SET NULL,
    rating        SMALLINT         NOT NULL,
    reason        feedback_reason,
    comment       TEXT,
    answer_source message_source   NOT NULL,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rating_values
        CHECK (rating IN (-1, 1))
);

COMMENT ON TABLE  feedback IS 'Оценки пользователей (👍 = 1, 👎 = -1)';
COMMENT ON COLUMN feedback.answer_source IS 'Источник ответа, который оценивался (faq/rag/api/operator)';

CREATE INDEX idx_feedback_session ON feedback (session_id);
CREATE INDEX idx_feedback_rating  ON feedback (rating);
CREATE INDEX idx_feedback_created ON feedback (created_at);
CREATE INDEX idx_feedback_source  ON feedback (answer_source);


-- ────────────────────────────────────────────────────────────
-- 8. ЭСКАЛАЦИИ
-- ────────────────────────────────────────────────────────────

CREATE TABLE escalations (
    id                SERIAL             PRIMARY KEY,
    session_id        INTEGER            NOT NULL
                                         REFERENCES sessions(id)
                                         ON DELETE CASCADE,
    reason            escalation_reason  NOT NULL,
    trigger_type      escalation_trigger NOT NULL,
    detected_category VARCHAR(50),
    operator_id       VARCHAR(50),
    queued_at         TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
    accepted_at       TIMESTAMPTZ,
    resolved_at       TIMESTAMPTZ,
    context_snapshot  JSONB              NOT NULL,

    CONSTRAINT chk_accepted_after_queued
        CHECK (accepted_at IS NULL
               OR accepted_at >= queued_at),
    CONSTRAINT chk_resolved_after_accepted
        CHECK (resolved_at IS NULL
               OR resolved_at >= accepted_at)
);

COMMENT ON TABLE  escalations IS 'Эскалации диалогов к операторам';
COMMENT ON COLUMN escalations.context_snapshot IS 'Снимок контекста сессии на момент эскалации: {history: [...], category: ..., order_number: ...}';
COMMENT ON COLUMN escalations.operator_id IS 'ID оператора из JAICP LiveChat';

CREATE INDEX idx_escalations_session  ON escalations (session_id);
CREATE INDEX idx_escalations_reason   ON escalations (reason);
CREATE INDEX idx_escalations_queued   ON escalations (queued_at);
CREATE INDEX idx_escalations_operator ON escalations (operator_id)
    WHERE operator_id IS NOT NULL;


-- ────────────────────────────────────────────────────────────
-- 9. КОНФИГУРАЦИЯ
-- ────────────────────────────────────────────────────────────

CREATE TABLE system_config (
    id          SERIAL           PRIMARY KEY,
    key         VARCHAR(100)     NOT NULL UNIQUE,
    value       TEXT             NOT NULL,
    value_type  config_value_type NOT NULL DEFAULT 'string',
    description TEXT,
    updated_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_by  VARCHAR(50)
);

COMMENT ON TABLE system_config IS 'Системные настройки (пороги, лимиты, флаги)';

CREATE INDEX idx_config_key ON system_config (key);


-- ────────────────────────────────────────────────────────────
-- 10. АВТООБНОВЛЕНИЕ updated_at
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Применение триггера к мутабельным таблицам
CREATE TRIGGER trg_categories_updated
    BEFORE UPDATE ON categories
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

CREATE TRIGGER trg_faq_updated
    BEFORE UPDATE ON faq_entries
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

CREATE TRIGGER trg_kb_documents_updated
    BEFORE UPDATE ON kb_documents
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

CREATE TRIGGER trg_orders_updated
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();


-- ────────────────────────────────────────────────────────────
-- 11. АВТОИНКРЕМЕНТ message_count в sessions
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION fn_increment_message_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE sessions
    SET message_count = message_count + 1
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_messages_count
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION fn_increment_message_count();


-- ────────────────────────────────────────────────────────────
-- 12. АВТОИНКРЕМЕНТ hit_count в faq_entries
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION fn_increment_faq_hit(p_faq_id INTEGER)
RETURNS VOID AS $$
BEGIN
    UPDATE faq_entries
    SET hit_count = hit_count + 1
    WHERE id = p_faq_id;
END;
$$ LANGUAGE plpgsql;