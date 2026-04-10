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
    $global.API_BASE_URL = "https://74c8018b-d87c-4adc-ac88-071708caed89.tunnel4.com";
    $global.API_KEY = "your-secret-key-here";  
    # Добавь в .env потом
    
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
            // Инициализация сессии
            $session.failCount = 0;
            $session.category = null;
            $session.confidence = 0;
            $session.orderNumber = null;
            $session.dialogHistory = [];
        a: Переходим к приветствию
        go!: /Greeting/Welcome

    state: CatchAll
        event!: noMatch
        a: Перехватываем все необработанные сообщения
        script:
            $reactions.answer("DEBUG CatchAll: " + $request.query);
        go!: /ClassifierRouter/Route
