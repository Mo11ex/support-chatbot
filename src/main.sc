require: faq.sc
require: orderStatus.sc
require: escalation.sc
require: feedback.sc

# Настройки API
init:
    $session.API_URL = "https://ТВОЙ_NGROK_URL.ngrok-free.app";
    $session.failCount = 0;
    $session.orderNumber = null;
    $session.lastCategory = null;

# ══════════════════════════════════════════════
# КОРНЕВОЕ СОСТОЯНИЕ
# ══════════════════════════════════════════════

theme: /

    state: Start
        q!: $regex</start>
        q!: привет
        q!: здравствуйте
        q!: добрый день
        q!: добрый вечер
        q!: хай
        q!: hello
        q!: hi
        a: Здравствуйте! 👋 Я бот службы поддержки интернет-магазина.
        a: Чем могу помочь?
        buttons:
            "📦 Статус заказа" -> /OrderFlow/AskOrderNumber
            "🔄 Возврат товара" -> /FAQ/ReturnInfo
            "🚚 Доставка" -> /FAQ/DeliveryInfo
            "💳 Оплата" -> /FAQ/PaymentInfo
            "❓ Другой вопрос" -> /Router/Process
            "👨‍💼 Оператор" -> /Escalation/Connect

    # ══════════════════════════════════════════════
    # МЕТА-ИНТЕНТЫ (обрабатываются в любом месте)
    # ══════════════════════════════════════════════

    state: Help
        q!: помощь
        q!: помоги
        q!: что ты умеешь
        q!: чем можешь помочь
        q!: меню
        a: Я могу помочь с:
        buttons:
            "📦 Статус заказа" -> /OrderFlow/AskOrderNumber
            "🔄 Возврат товара" -> /FAQ/ReturnInfo
            "🚚 Доставка" -> /FAQ/DeliveryInfo
            "💳 Оплата" -> /FAQ/PaymentInfo
            "🏷️ Промокоды" -> /FAQ/PromoInfo
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: Thanks
        q!: спасибо
        q!: благодарю
        q!: спс
        q!: thanks
        a: Рад помочь! 😊 Есть ещё вопросы?
        buttons:
            "Да, есть вопрос" -> /Start
            "Нет, спасибо" -> /Goodbye

    state: Goodbye
        q!: пока
        q!: до свидания
        q!: всё спасибо
        q!: нет спасибо
        q!: bye
        a: Спасибо за обращение! Если будут вопросы — пишите. До свидания! 👋

    state: OperatorRequest
        q!: оператор
        q!: позовите оператора
        q!: живой человек
        q!: соедините с оператором
        q!: хочу поговорить с человеком
        q!: менеджер
        q!: позовите менеджера
        go!: /Escalation/Connect

# ══════════════════════════════════════════════
# РОУТЕР — классификация через ML API
# ══════════════════════════════════════════════

theme: /Router

    state: Process
        go!: /FAQ/SearchFAQ

    state: Classify
        script:
            var query = $request.query;

            try {
                var resp = $http.post($session.API_URL + "/api/v1/classify", {
                    headers: {"Content-Type": "application/json"},
                    body: {"text": query, "session_id": $request.channelUserId},
                    timeout: 5000
                });

                if (resp.status == 200) {
                    $session.lastCategory = resp.body.category;
                    $session.lastConfidence = resp.body.confidence;
                    $session.failCount = 0;
                } else {
                    $session.lastCategory = "error";
                    $session.lastConfidence = 0;
                }
            } catch(e) {
                $session.lastCategory = "error";
                $session.lastConfidence = 0;
            }

        # Высокая уверенность — маршрутизация
        if: $session.lastCategory == "order_status" && $session.lastConfidence >= 0.65
            go!: /OrderFlow/AskOrderNumber

        elseif: $session.lastCategory == "return_exchange" && $session.lastConfidence >= 0.65
            go!: /FAQ/ReturnInfo

        elseif: $session.lastCategory == "delivery" && $session.lastConfidence >= 0.65
            go!: /FAQ/DeliveryInfo

        elseif: $session.lastCategory == "payment_refund" && $session.lastConfidence >= 0.65
            go!: /FAQ/RefundInfo

        elseif: $session.lastCategory == "product_info" && $session.lastConfidence >= 0.65
            a: К сожалению, я пока не могу подробно рассказать о товарах. Хотите связаться с оператором?
            buttons:
                "👨‍💼 Оператор" -> /Escalation/Connect
                "🏠 Главное меню" -> /Start

        elseif: $session.lastCategory == "account" && $session.lastConfidence >= 0.65
            go!: /FAQ/PasswordInfo

        elseif: $session.lastCategory == "promo_loyalty" && $session.lastConfidence >= 0.65
            go!: /FAQ/PromoInfo

        elseif: $session.lastCategory == "technical_issue" && $session.lastConfidence >= 0.65
            a: 🔧 Для решения технических проблем лучше обратиться к специалисту.
            go!: /Escalation/Connect

        elseif: $session.lastCategory == "general_info" && $session.lastConfidence >= 0.65
            go!: /FAQ/ContactsInfo

        elseif: $session.lastConfidence >= 0.40
            # Средняя уверенность — уточнение
            a: Уточните, пожалуйста, ваш вопрос:
            buttons:
                "📦 Статус заказа" -> /OrderFlow/AskOrderNumber
                "🔄 Возврат товара" -> /FAQ/ReturnInfo
                "🚚 Доставка" -> /FAQ/DeliveryInfo
                "💳 Оплата" -> /FAQ/PaymentInfo
                "🏷️ Промокоды" -> /FAQ/PromoInfo
                "👨‍💼 Оператор" -> /Escalation/Connect

        else:
            # Низкая уверенность или ошибка
            script:
                $session.failCount = ($session.failCount || 0) + 1;

            if: $session.failCount >= 2
                a: Похоже, я не могу помочь с этим вопросом. Давайте подключу оператора.
                go!: /Escalation/Connect
            else:
                a: Не совсем понял ваш вопрос 🤔 Попробуйте переформулировать или выберите тему:
                buttons:
                    "📦 Статус заказа" -> /OrderFlow/AskOrderNumber
                    "🔄 Возврат товара" -> /FAQ/ReturnInfo
                    "🚚 Доставка" -> /FAQ/DeliveryInfo
                    "💳 Оплата" -> /FAQ/PaymentInfo
                    "👨‍💼 Оператор" -> /Escalation/Connect