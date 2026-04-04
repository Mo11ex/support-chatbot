# ══════════════════════════════════════════════
# СЦЕНАРИЙ СТАТУСА ЗАКАЗА
# ══════════════════════════════════════════════

theme: /OrderFlow

    state: AskOrderNumber
        q!: где мой заказ
        q!: статус заказа
        q!: отследить заказ
        q!: где посылка
        q!: когда придёт заказ
        q!: когда доставят
        q!: что с моим заказом
        q!: дайте трек номер
        a: Введите номер заказа (например, 100001):

    state: ProcessOrder
        event!: noMatch
        script:
            var userInput = $request.query.trim();

            // Извлекаем число из текста
            var match = userInput.match(/\d{5,8}/);

            if (match) {
                $session.orderNumber = match[0];

                try {
                    var resp = $http.get($session.API_URL + "/api/v1/orders/" + $session.orderNumber, {
                        headers: {"Content-Type": "application/json"},
                        timeout: 5000
                    });

                    if (resp.status == 200) {
                        var order = resp.body;
                        $session.orderFound = true;
                        $session.orderStatus = order.status_label;
                        $session.orderDelivery = order.estimated_delivery || "не указана";
                        $session.orderTrack = order.tracking_number || "не присвоен";
                        $session.orderAmount = order.total_amount;
                        $session.orderItems = order.items;
                    } else {
                        $session.orderFound = false;
                    }
                } catch(e) {
                    $session.orderFound = false;
                }
            } else {
                $session.orderNumber = null;
                $session.orderFound = false;
            }

        if: $session.orderNumber == null
            a: ❌ Не удалось найти номер заказа в вашем сообщении. Введите номер (например, 100001):

        elseif: $session.orderFound
            a: 📦 Заказ #{{$session.orderNumber}}
            a: Статус: {{$session.orderStatus}}
            a: Ожидаемая доставка: {{$session.orderDelivery}}
            a: Трек-номер: {{$session.orderTrack}}
            a: Сумма: {{$session.orderAmount}}₽
            buttons:
                "📦 Другой заказ" -> /OrderFlow/AskOrderNumber
                "👍 Помогло" -> /Feedback/Positive
                "🏠 Главное меню" -> /Start
                "👨‍💼 Оператор" -> /Escalation/Connect

        else:
            a: ❌ Заказ #{{$session.orderNumber}} не найден.
            a: Проверьте номер и попробуйте ещё раз.
            buttons:
                "🔄 Ввести номер заново" -> /OrderFlow/AskOrderNumber
                "👨‍💼 Оператор" -> /Escalation/Connect
                "🏠 Главное меню" -> /Start