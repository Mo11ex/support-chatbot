theme: /ClassifierRouter

    state: Route
        script:
            var userText = $request.query;
            
            # Добавляем в историю
            if (!$session.dialogHistory) {
                $session.dialogHistory = [];
            }
            $session.dialogHistory.push({
                "user": userText,
                "timestamp": new Date().toISOString()
            });
            
            # ────────────────────────────────────────
            # ШАГ 1: Попытка FAQ Fast-Path
            # ────────────────────────────────────────
            
            try {
                var faqUrl = $global.API_BASE_URL + "/faq/match";
                var faqResp = $http.post(faqUrl, {
                    headers: {
                        "Content-Type": "application/json",
                        "X-API-Key": $global.API_KEY
                    },
                    body: {
                        "text": userText,
                        "session_id": $session.id
                    },
                    timeout: 2000
                });
                
                if (faqResp && faqResp.status == 200 && faqResp.body.matched) {
                    log("FAQ matched: " + faqResp.body.faq_id);
                    $session.lastSource = "faq";
                    $reactions.answer(faqResp.body.answer);
                    $reactions.buttons([
                        {text: "✅ Да, спасибо", transition: "/Greeting/Help"},
                        {text: "❌ Нет, другой вопрос", transition: "/ClassifierRouter/Route"}
                    ]);
                    $session.failCount = 0;
                    return;
                }
            } catch(e) {
                log("FAQ service error: " + e.message);
                # Продолжаем к классификатору
            }
            
            # ────────────────────────────────────────
            # ШАГ 2: ML-Классификатор
            # ────────────────────────────────────────
            
            try {
                var clsUrl = $global.API_BASE_URL + "/classify";
                var clsResp = $http.post(clsUrl, {
                    headers: {
                        "Content-Type": "application/json",
                        "X-API-Key": $global.API_KEY
                    },
                    body: {
                        "text": userText,
                        "session_id": $session.id
                    },
                    timeout: 3000
                });
                
                if (clsResp && clsResp.status == 200) {
                    $session.category = clsResp.body.category;
                    $session.confidence = clsResp.body.confidence;
                    
                    log("Classified: " + $session.category + " (conf: " + $session.confidence + ")");
                    
                    # ── Высокая уверенность ──
                    if ($session.confidence >= $global.CLASSIFIER_THRESHOLD_HIGH) {
                        $session.failCount = 0;
                        go!: /ClassifierRouter/RouteByCategory
                    }
                    
                    # ── Средняя уверенность → уточнение ──
                    else if ($session.confidence >= $global.CLASSIFIER_THRESHOLD_MID) {
                        go!: /ClassifierRouter/AskClarification
                    }
                    
                    # ── Низкая уверенность → эскалация ──
                    else {
                        $session.failCount++;
                        go!: /Fallback
                    }
                }
                
            } catch(e) {
                log("Classifier API error: " + e.message);
                go!: /ClassifierRouter/FallbackMenu
            }

    # ────────────────────────────────────────────────────────
    # Маршрутизация по категории
    # ────────────────────────────────────────────────────────
    
    state: RouteByCategory
        script:
            var category = $session.category;
            
            switch(category) {
                case "order_status":
                    $reactions.transition("/OrderStatus/AskOrderNumber");
                    break;
                    
                case "payment_refund":
                case "return_exchange":
                case "delivery":
                case "product_info":
                case "general_info":
                case "promo_loyalty":
                    # Эти категории идут в RAG (реализуем позже)
                    $reactions.answer("Сейчас ищу информацию по вашему вопросу...");
                    # TODO: go!: /RAGHandler/Query
                    $reactions.answer("⚠️ RAG временно недоступен. Соединяю с оператором...");
                    $reactions.transition("/Escalation/RequestOperator");
                    break;
                    
                case "account":
                case "technical_issue":
                    # Эти категории → сразу оператор
                    $reactions.transition("/Escalation/AutoEscalate");
                    break;
                    
                case "other":
                default:
                    $reactions.transition("/Fallback");
            }

    # ────────────────────────────────────────────────────────
    # Уточнение при средней уверенности
    # ────────────────────────────────────────────────────────
    
    state: AskClarification
        a: Уточните, пожалуйста, по какой теме ваш вопрос:
        script:
            var top3 = $session.top3 || [];
            var buttons = [];
            
            # Формируем кнопки из top-3 классификатора
            # (В реальности берём из clsResp.body.top_3)
            buttons.push({text: "📦 Статус заказа", transition: "/OrderStatus/AskOrderNumber"});
            buttons.push({text: "💳 Оплата и возврат", transition: "/ClassifierRouter/Route"});
            buttons.push({text: "📞 Связаться с оператором", transition: "/Escalation/RequestOperator"});
            
            $reactions.buttons(buttons);

    # ────────────────────────────────────────────────────────
    # Fallback при ошибке API
    # ────────────────────────────────────────────────────────
    
    state: FallbackMenu
        a: Выберите тему обращения:
        buttons:
            "📦 Статус заказа" -> /OrderStatus/AskOrderNumber
            "❓ Частые вопросы" -> /FAQHandler/ShowMenu
            "💬 Связаться с оператором" -> /Escalation/RequestOperator