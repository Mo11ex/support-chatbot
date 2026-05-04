# Support Chatbot

Умный чат-бот для интернет-магазина на платформе JustAI (JAICP) с RAG-архитектурой.

---

## Структура репозитория

support-chatbot/
├── backend/ # FastAPI backend, API endpoints, ML inference
├── src/ # JAICP сценарии и диалоговые состояния
├── ml/ # ML-исследования: данные, обучение, оценка
│ ├── data/
│ │ ├── raw/ # исходные датасеты
│ │ ├── interim/ # промежуточные данные
│ │ ├── processed/ # финальные датасеты и корпуса
│ │ ├── splits/ # train / val / test
│ │ └── heldout/ # held-out benchmark 200 запросов
│ ├── models/ # обученные модели ли артефакты
│ ├── src/
│ │ └── evaluation/ # модуль метрик
│ ├── notebooks/ # исследовательские ноутбуки
│ ├── scripts/ # воспроизводимые скрипты
│ ├── configs/ # конфигурации и маппинги
│ └── logs/ # отчёты и артефакты экспериментов
├── docs/ # документация
└── README.md


---

## Установка

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Сборка единого датасета
backend\venv\Scripts\python.exe ml\scripts\build_full_dataset.py

# Разбивка на train/val/test
backend\venv\Scripts\python.exe ml\scripts\make_splits.py

# Проверка сплитов
backend\venv\Scripts\python.exe ml\scripts\check_splits.py

# Сборка FAQ корпуса
backend\venv\Scripts\python.exe ml\scripts\build_faq_corpus.py

# Smoke test модуля метрик
backend\venv\Scripts\python.exe ml\scripts\smoke_test_evaluation.py

# Запуск MLflow UI
mlflow ui
# Открыть http://127.0.0.1:5000

Данные
Файл	Описание
ml/data/processed/full_dataset.csv	Единый нормализованный датасет (3038 записей)
ml/data/splits/train.csv	Train выборка (2126 записей, 70%)
ml/data/splits/val.csv	Validation выборка (456 записей, 15%)
ml/data/splits/test.csv	Test выборка (456 записей, 15%)
ml/data/processed/faq_corpus.csv	FAQ корпус (59 чанков, 5 документов)
ml/data/processed/faq_eval.csv	FAQ retrieval benchmark (63 запроса)
ml/data/heldout/heldout_200.csv	End-to-end benchmark (200 запросов)
Схема интентов
10 классов, random_seed=42:

label	Описание	route
account	Аккаунт, пароль, регистрация	fallback
delivery	Доставка, сроки, стоимость	faq_handler
general_info	Общая информация о магазине	fallback
order_status	Статус и трекинг заказа	order_status
other	Вне домена, шум	fallback
payment_refund	Оплата и возврат средств	faq_handler
product_info	Информация о товаре	fallback
promo_loyalty	Промокоды, акции, бонусы	faq_handler
return_exchange	Возврат и обмен товара	faq_handler
technical_issue	Технические проблемы	escalation
Целевые метрики проекта
Метрика	Цель
Классификатор macro F1	≥ 0.88
Классификатор F1 по каждому классу	≥ 0.80
FAQ retrieval precision@1	≥ 0.90
FAQ retrieval recall@5	≥ 0.85
RAG recall@5	≥ 0.85
End-to-end accuracy на held-out	≥ 0.85
Latency P95 на CPU	≤ 1.5 сек
Метрики считаются на:

классификатор — ml/data/splits/test.csv
retrieval — ml/data/processed/faq_eval.csv
end-to-end — ml/data/heldout/heldout_200.csv
Воспроизводимость
Все результаты воспроизводятся запуском скриптов из корня репозитория.
random_seed=42 зафиксирован во всех скриптах.
Сплиты зафиксированы и не пересоздаются автоматически.


---

# Шаг: MLflow

## Установить

```powershell
backend\venv\Scripts\python.exe -m pip install mlflow

backend\venv\Scripts\python.exe -c "import mlflow; print(mlflow.__version__)"


