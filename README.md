# Support Chatbot

Умный чат-бот для интернет-магазина на платформе JustAI (JAICP) с RAG-архитектурой.

Функциональность:
- Классификация пользовательских запросов (10 интентов)
- Ответы на вопросы по базе знаний (RAG)
- Интеграция с данными заказов
- Переключение на оператора
- Fallback для нераспознанных запросов

---

## Структура репозитория

```
support-chatbot/
├── backend/                        # FastAPI backend
│   ├── app/
│   │   ├── api/v1/                 # API endpoints
│   │   ├── ml/                     # Runtime ML модели
│   │   │   ├── classifier/         # Классификатор интентов
│   │   │   ├── embedder/           # Эмбеддер для RAG
│   │   │   └── rag/                # RAG индекс
│   │   └── services/               # Бизнес-логика
│   ├── data/
│   │   └── knowledge_base/         # Документы базы знаний
│   └── requirements.txt
│
├── src/                            # JAICP сценарии
│   └── states/
│       ├── greeting.sc
│       ├── order_status.sc
│       ├── faq_handler.sc
│       ├── escalation.sc
│       ├── fallback.sc
│       └── classifier_router.sc
│
├── ml/                             # ML исследования и обучение
│   ├── data/
│   │   ├── raw/                    # Исходные датасеты
│   │   ├── interim/                # Промежуточные данные
│   │   ├── processed/              # Финальные датасеты
│   │   │   ├── full_dataset.csv    # Единый нормализованный датасет
│   │   │   ├── faq_corpus.csv      # FAQ корпус (чанки)
│   │   │   └── faq_eval.csv        # FAQ retrieval benchmark
│   │   ├── splits/                 # Train / Val / Test
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   │   │   └── test.csv
│   │   └── heldout/
│   │       └── heldout_200.csv     # End-to-end benchmark
│   │
│   ├── models/                     # Обученные модели
│   │   ├── classifier/
│   │   ├── retriever/
│   │   └── artifacts/
│   │
│   ├── src/
│   │   ├── evaluation/
│   │   │   └── evaluation.py       # Модуль метрик
│   │   └── utils/
│   │       └── tracking.py         # MLflow утилиты
│   │
│   ├── notebooks/                  # Исследовательские ноутбуки
│   ├── scripts/                    # Воспроизводимые скрипты
│   │   ├── build_full_dataset.py
│   │   ├── build_faq_corpus.py
│   │   ├── make_splits.py
│   │   ├── check_splits.py
│   │   ├── smoke_test_evaluation.py
│   │   ├── smoke_test_mlflow.py
│   │   └── inspect_datasets.py
│   │
│   ├── configs/
│   │   └── intent_mapping.yaml     # Схема интентов и маппинг
│   │
│   └── logs/                       # Отчёты и MLflow артефакты
│       └── mlflow.db
│
├── docs/
├── .gitignore
└── README.md
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Запуск backend

```bash
cd backend
uvicorn app.main:app --reload
```

### 3. Запуск MLflow UI

```bash
# Из корня support-chatbot
backend\venv\Scripts\python.exe -m mlflow ui \
  --backend-store-uri sqlite:///ml/logs/mlflow.db \
  --default-artifact-root ml/logs/mlruns
```

Открыть: http://127.0.0.1:5000

---

## Воспроизводимость данных

Все скрипты запускаются из корня `support-chatbot`.

```bash
# Сборка единого датасета
backend\venv\Scripts\python.exe ml\scripts\build_full_dataset.py

# Разбивка на train/val/test (70/15/15, seed=42)
backend\venv\Scripts\python.exe ml\scripts\make_splits.py

# Проверка сплитов
backend\venv\Scripts\python.exe ml\scripts\check_splits.py

# Сборка FAQ корпуса из knowledge base
backend\venv\Scripts\python.exe ml\scripts\build_faq_corpus.py

# Smoke test модуля метрик
backend\venv\Scripts\python.exe ml\scripts\smoke_test_evaluation.py

# Smoke test MLflow
backend\venv\Scripts\python.exe ml\scripts\smoke_test_mlflow.py
```

---

## Данные

| Файл | Описание | Размер |
|---|---|---|
| `ml/data/processed/full_dataset.csv` | Единый нормализованный датасет | 3038 записей |
| `ml/data/splits/train.csv` | Train выборка | 2126 (70%) |
| `ml/data/splits/val.csv` | Validation выборка | 456 (15%) |
| `ml/data/splits/test.csv` | Test выборка | 456 (15%) |
| `ml/data/processed/faq_corpus.csv` | FAQ корпус | 59 чанков, 5 документов |
| `ml/data/processed/faq_eval.csv` | FAQ retrieval benchmark | 63 запроса |
| `ml/data/heldout/heldout_200.csv` | End-to-end benchmark | 220 запросов |

Источники датасета:

| Источник | Записей | Синтетика |
|---|---|---|
| bitext | 1862 | Нет |
| ikshana | 823 | Нет |
| runlu | 218 | Нет |
| synthetic_dataset | 135 | Да |

---

## Схема интентов

10 классов, `random_seed=42`, стратифицированное разбиение.

| label | Описание | route | KB документ |
|---|---|---|---|
| `account` | Аккаунт, пароль, регистрация | fallback | — |
| `delivery` | Доставка, сроки, стоимость | faq_handler | delivery_policy.md |
| `general_info` | Общая информация о магазине | fallback | — |
| `order_status` | Статус и трекинг заказа | order_status | — |
| `other` | Вне домена, шум | fallback | — |
| `payment_refund` | Оплата и возврат средств | faq_handler | payment_methods.md |
| `product_info` | Информация о товаре | fallback | — |
| `promo_loyalty` | Промокоды, акции, бонусы | faq_handler | promo_codes.md |
| `return_exchange` | Возврат и обмен товара | faq_handler | return_policy.md |
| `technical_issue` | Технические проблемы | escalation | — |

---

## Целевые метрики проекта

| Метрика | Цель | Измеряется на |
|---|---|---|
| Классификатор macro F1 | ≥ 0.88 | `test.csv` |
| Классификатор F1 по каждому классу | ≥ 0.80 | `test.csv` |
| FAQ retrieval precision@1 | ≥ 0.90 | `faq_eval.csv` |
| FAQ retrieval recall@5 | ≥ 0.85 | `faq_eval.csv` |
| RAG recall@5 | ≥ 0.85 | `faq_eval.csv` |
| End-to-end accuracy | ≥ 0.85 | `heldout_200.csv` |
| Latency P95 на CPU | ≤ 1.5 сек | inference benchmark |

---

## Трекинг экспериментов

Все эксперименты логируются в MLflow.

```
ml/logs/mlflow.db       — база данных экспериментов
ml/logs/mlruns/         — артефакты моделей
```

Логируются:
- гиперпараметры модели
- метрики (macro F1, accuracy, precision@k, recall@k, MRR)
- confusion matrix
- classification report
- артефакты моделей

---

## Воспроизводимость

- `random_seed = 42` зафиксирован во всех скриптах
- Сплиты зафиксированы и не пересоздаются автоматически
- Все результаты воспроизводятся запуском скриптов из корня репозитория
- Зависимости зафиксированы в `backend/requirements.txt`

---

## Известные ограничения

- Класс `warranty` не выделен как отдельный интент в классификаторе.
  Вопросы о гарантии попадают в смежные категории.
  FAQ retrieval для warranty работает через документ `warranty.md`.

- Датасет содержит записи на английском языке (источник: bitext).
  Часть записей переведена автоматически. Возможны артефакты перевода.

- Классы `payment_refund` и `return_exchange` семантически близки,
  что может негативно влиять на качество классификации.

---

## Этапы разработки

- [x] Этап 0 — Инфраструктура, данные, метрики
- [ ] Этап 1 — Классификатор интентов
- [ ] Этап 2 — FAQ Retrieval
- [ ] Этап 3 — RAG pipeline
- [ ] Этап 4 — Интеграция с JAICP
- [ ] Этап 5 — End-to-end тестирование