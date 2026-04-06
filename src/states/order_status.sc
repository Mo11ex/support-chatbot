theme: /OrderStatus

    state: AskOrderNumber
        a: Введите номер вашего заказа (например, 100001):
        
    state: ProcessOrder
        q: * $Number *
        script:
            var orderNum = $parseTree._Number;
            
            # Валидация формата
            if (!orderNum || orderNum.toString().length < 4) {
                $reactions.answer("Номер заказа должен содержать минимум 4 цифры. Попробуйте ещё раз:");
                return;
            }
            
            $session.orderNumber = orderNum.toString();
            
            # Запрос к API
            try {
                var orderUrl = $global.API_BASE_URL + "/orders/" + $session.orderNumber;
                var orderResp = $http.get(orderUrl, {
                    headers: {
                        "X-API-Key": $global.API_KEY
                    },
                    timeout: 3000
                });
                
                if (orderResp && orderResp.status == 200) {
                    var order = orderResp.body;
                    
                    var msg = "📦 Заказ #" + order.order_number + "\n";
                    msg += "Статус: " + order.status_label + "\n";
                    msg += "Дата создания: " + order.created_at.substring(0,10) + "\n";
                    
                    if (order.estimated_delivery) {
                        msg += "Ожидаемая доставка: " + order.estimated_delivery + "\n";
                    }
                    
                    if (order.tracking_number) {
                        msg += "Трек-номер: " + order.tracking_number + "\n";
                    }
                    
                    msg += "Товаров: " + order.items_count + "\n";
                    msg += "Сумма: " + order.total_amount + " ₽";
                    
                    $reactions.answer(msg);
                    $reactions.buttons([
                        {text: "🔍 Другой заказ", transition: "/OrderStatus/AskOrderNumber"},
                        {text: "🏠 Главное меню", transition: "/Greeting/Help"}
                    ]);
                    
                    $session.failCount = 0;
                    
                } else if (orderResp && orderResp.status == 404) {
                    go!: /OrderStatus/NotFound
                }
                
            } catch(e) {
                log("Orders API error: " + e.message);
                go!: /OrderStatus/APIError
            }
    
    state: NotFound
        a: Заказ с номером {{$session.orderNumber}} не найден 😔
        a: Проверьте номер или обратитесь к оператору:
        buttons:
            "🔄 Ввести номер заново" -> /OrderStatus/AskOrderNumber
            "💬 Связаться с оператором" -> /Escalation/RequestOperator
    
    state: APIError
        a: Не удалось получить информацию о заказе. Попробуйте позже или обратитесь к оператору:
        buttons:
            "🔄 Попробовать ещё раз" -> /OrderStatus/AskOrderNumber
            "💬 Связаться с оператором" -> /Escalation/RequestOperator