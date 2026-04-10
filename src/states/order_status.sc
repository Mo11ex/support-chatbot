theme: /OrderStatus

    state: AskOrderNumber
        #TODO: intent: и сущность
        a: Введите номер вашего заказа (например, 100001):
        
    state: ProcessOrder
        q: * @duckling.number *
        script:
            var orderNum = $parseTree.value;
            
            if (!orderNum || orderNum.toString().length < 4) {
                $reactions.answer("Номер заказа должен содержать минимум 4 цифры.");
                return;
            }
            
            $session.orderNumber = orderNum.toString();
            $reactions.answer("Ищу заказ № " + $session.orderNumber);
            
            try {
                var orderUrl = $global.API_BASE_URL + "/api/v1/orders/" + $session.orderNumber;
                
                var orderResp = $http.query(orderUrl, {
                    method: "GET",
                    headers: {
                        "X-API-Key": $global.API_KEY
                    },
                    timeout: 5000,
                    dataType: "text"
                });
                
                if (orderResp && orderResp.status == 200) {
                    // Парсим JSON вручную
                    var order = JSON.parse(orderResp.data);
                    
                    var msg = "📦 Заказ #" + order.order_number + "\n";
                    msg += "Статус: " + order.status_label + "\n";
                    
                    if (order.created_at) {
                        msg += "Дата создания: " + order.created_at.substring(0, 10) + "\n";
                    }
                    
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
                    $reactions.transition("/OrderStatus/NotFound");
                }
                
            } catch(e) {
                $reactions.answer("Ошибка: " + e.message);
                log("Orders API error: " + e.message);
                $reactions.transition("/OrderStatus/APIError");
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