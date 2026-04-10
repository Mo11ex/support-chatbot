theme: /Greeting

    state: Welcome
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

    state: TestAPI
        q: тест апи
        script:
            try {
                var url = $global.API_BASE_URL + "/orders/100001";
                $reactions.answer("URL: " + url);
                
                var resp = $http.get(url, {
                    headers: {"X-API-Key": $global.API_KEY},
                    timeout: 5000
                });
                
                $reactions.answer("Status: " + resp.status);
                $reactions.answer("Response: " + JSON.stringify(resp).substring(0, 1000));
                
            } catch(e) {
                $reactions.answer("Error: " + e.message);
            }