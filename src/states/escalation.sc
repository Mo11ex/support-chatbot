theme: /Escalation

    state: RequestOperator
        q!: * (оператор|живой человек|менеджер|позовите|хочу с человеком) *
        a: Соединяю с оператором...
        a: 
        a: ⏳ Пожалуйста, ожидайте. Среднее время ответа: 2-3 минуты.
        script:
            # Передаём контекст оператору
            var context = {
                "category": $session.category || "unknown",
                "orderNumber": $session.orderNumber || null,
                "failCount": $session.failCount || 0,
                "history": $session.dialogHistory || []
            };
            
            log("Escalation context: " + JSON.stringify(context));
            
            # TODO: Интеграция с JAICP LiveChat
            # $reactions.transferToOperator(context);
        
        # Заглушка для MVP
        a: ⚠️ LiveChat временно недоступен.
        a: Оставьте ваш email, мы свяжемся с вами в течение часа:
        # TODO: собрать email
    
    state: AutoEscalate
        script:
            log("Auto-escalation triggered. Reason: " + ($session.escalationReason || "max_failures"));
        a: Передаю ваше обращение оператору...
        go!: /Escalation/RequestOperator