require: states/greeting.sc
require: states/classifier_router.sc
require: states/order_status.sc
require: states/faq_handler.sc
require: states/fallback.sc
require: states/escalation.sc

# ════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════

init:
    # URL твоего FastAPI (замени на реальный при деплое)
    $global.API_BASE_URL = "http://localhost:8000/api/v1";
    $global.API_KEY = "your-secret-key-here";  # Добавь в .env потом
    
    # Пороги
    $global.CLASSIFIER_THRESHOLD_HIGH = 0.65;
    $global.CLASSIFIER_THRESHOLD_MID = 0.40;
    $global.MAX_FAIL_COUNT = 2;

# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

theme: /

    state: Start
        q!: $regex</start>
        script:
            # Инициализация сессии
            $session.failCount = 0;
            $session.category = null;
            $session.confidence = 0;
            $session.orderNumber = null;
            $session.dialogHistory = [];
        a: Переходим к приветствию
        go!: /Greeting

    state: CatchAll
        event!: noMatch
        a: Перехватываем все необработанные сообщения
        go!: /ClassifierRouter

    state: NoMatch
        event!: noMatch
        a: Что-то пошло не так
        go!: /Fallback