theme: /Greeting

    state: Welcome
        #TODO: intent:
        q!: * (привет|здравствуй|добрый день|hi|hello) *
        q!: $regex</start>
        script:
            $session.failCount = 0;
        a: Здравствуйте! 👋 Я виртуальный помощник интернет-магазина.
        a: Чем могу помочь?
        buttons:
            "📦 Статус заказа" -> /OrderStatus/AskOrderNumber
            "❓ Частые вопросы" -> /FAQHandler/ShowMenu
            "💬 Связаться с оператором" -> /Escalation/RequestOperator

    state: Help
        #TODO: intent:
        q!: * (помощь|что ты умеешь|меню|menu|help) *
        a: Я могу помочь с:
         • Отслеживанием заказов
         • Ответами на частые вопросы
         • Поиском информации в базе знаний
         • Связью с оператором
         
         Просто напишите ваш вопрос или выберите тему:
        buttons:
            "📦 Статус заказа" -> /OrderStatus/AskOrderNumber
            "❓ Частые вопросы" -> /FAQHandler/ShowMenu
            "💬 Связаться с оператором" -> /Escalation/RequestOperator
            
    #TODO: intent: Благодарность