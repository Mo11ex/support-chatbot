theme: /Fallback

    state: Handle
        script:
            $session.failCount = ($session.failCount || 0) + 1;
            
            log("Fallback triggered. Fail count: " + $session.failCount);
            
            if ($session.failCount >= $global.MAX_FAIL_COUNT) {
                go!: /Fallback/AutoEscalate
            }
        
        a: Извините, я не совсем понял ваш вопрос 🤔
        a: Попробуйте переформулировать или выберите тему:
        buttons:
            "📦 Статус заказа" -> /OrderStatus/AskOrderNumber
            "❓ Частые вопросы" -> /FAQHandler/ShowMenu
            "💬 Связаться с оператором" -> /Escalation/RequestOperator
    
    state: AutoEscalate
        a: Похоже, я не могу помочь с этим вопросом 😔
        a: Давайте подключу оператора:
        buttons:
            "✅ Подключить оператора" -> /Escalation/RequestOperator
            "🔄 Попробовать ещё раз" -> /ClassifierRouter/Route