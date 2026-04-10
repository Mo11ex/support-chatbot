theme: /ClassifierRouter

    state: Route
        script:
            var userText = ($request.query || "").trim();

            if (!userText) {
                $reactions.answer("Напишите вопрос текстом — я постараюсь помочь.");
                return;
            }

            // История (минимально)
            if (!$session.dialogHistory) $session.dialogHistory = [];
            $session.dialogHistory.push({ user: userText, ts: new Date().toISOString() });

            // Конфиги (дефолты, если $global не задан)
            var apiBase = $global.API_BASE_URL;
            var apiKey  = $global.API_KEY;
            var thrHigh = ($global.CLASSIFIER_THRESHOLD_HIGH != null) ? $global.CLASSIFIER_THRESHOLD_HIGH : 0.65;
            var thrMid  = ($global.CLASSIFIER_THRESHOLD_MID  != null) ? $global.CLASSIFIER_THRESHOLD_MID  : 0.40;

            // ───────────────────────────────
            // ШАГ 1: FAQ Fast-Path
            // ───────────────────────────────
            try {
                var faqResp = $http.query(apiBase + "/api/v1/faq/match", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": apiKey
                },
                body: { text: userText, session_id: $session.id },
                timeout: 2000
                });

                var faqStatus = faqResp.status || faqResp.statusCode;
                var faqData = faqResp.data || faqResp.body || {};

                if (faqStatus == 200 && faqData.matched) {
                $session.lastSource = "faq";
                $session.failCount = 0;

                $reactions.answer(faqData.answer);
                $reactions.buttons([
                    { text: "Задать другой вопрос", transition: "/ClassifierRouter/Route" }
                ]);
                return;
                }
            } catch (e) {
                log("FAQ error: " + e.message);
            }

            // ───────────────────────────────
            // ШАГ 2: Classify
            // ───────────────────────────────
            try {
                var clsResp = $http.query(apiBase + "/api/v1/classify", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": apiKey
                },
                body: { text: userText, session_id: $session.id },
                timeout: 3000
                });

                var clsStatus = clsResp.status || clsResp.statusCode;
                var clsData = clsResp.data || clsResp.body || {};

                if (clsStatus != 200 || !clsData.category) {
                log("Classify bad response, status=" + clsStatus);
                $reactions.transition("/ClassifierRouter/FallbackMenu");
                return;
                }

                $session.category = clsData.category;
                $session.confidence = clsData.confidence;
                $session.top3 = clsData.top_3 || [];
                $session.lastUserText = userText;

                // ── Высокая уверенность ──
                if ($session.confidence >= thrHigh) {
                $session.failCount = 0;
                $reactions.transition("/ClassifierRouter/RouteByCategory");
                return;
                }

                // ── Средняя уверенность ──
                if ($session.confidence >= thrMid) {
                $reactions.transition("/ClassifierRouter/AskClarification");
                return;
                }

                // ── Низкая уверенность ──
                // В MVP можно всё равно попытаться RAG (его similarity-threshold защитит от галлюцинаций)
                $session.failCount = ($session.failCount || 0) + 1;
                $session.ragQuery = userText;
                $reactions.transition("/RAGHandler/Query");
                return;

            } catch (e) {
                log("Classifier API error: " + e.message);
                $reactions.transition("/ClassifierRouter/FallbackMenu");
                return;
            }


    state: RouteByCategory
        script:
            var category = $session.category;

            switch (category) {
                case "order_status":
                $reactions.transition("/OrderStatus/AskOrderNumber");
                return;

                // Всё, кроме заказов — в RAG
                case "payment_refund":
                case "return_exchange":
                case "delivery":
                case "product_info":
                case "general_info":
                case "promo_loyalty":
                case "account":
                case "technical_issue":
                case "other":
                default:
                $session.ragQuery = $session.lastUserText || $request.query;
                $reactions.transition("/RAGHandler/Query");
                return;
            }


    state: AskClarification
        a: Уточните, пожалуйста, тему вопроса:
        script:
            // Сформируем кнопки по top-3, но переход сделаем в один обработчик
            var top3 = $session.top3 || [];

            // словарь code -> человеко-читаемый текст
            var label = {
                "order_status": "📦 Статус заказа",
                "payment_refund": "💳 Оплата и возврат",
                "return_exchange": "🔄 Возврат и обмен",
                "delivery": "🚚 Доставка",
                "product_info": "📱 Товар и ассортимент",
                "general_info": "ℹ️ Общая информация",
                "promo_loyalty": "🎁 Промокоды и бонусы",
                "account": "👤 Аккаунт",
                "technical_issue": "🔧 Тех. проблема",
                "other": "❓ Другое"
            };

            var buttons = [];
            for (var i = 0; i < top3.length && i < 3; i++) {
                var cat = top3[i].category;
                if (label[cat]) {
                buttons.push({ text: label[cat], transition: "/ClassifierRouter/ClarifiedRoute" });
                }
            }

            // запасные
            if (buttons.length < 2) {
                buttons = [
                { text: "📦 Статус заказа", transition: "/ClassifierRouter/ClarifiedRoute" },
                { text: "🚚 Доставка", transition: "/ClassifierRouter/ClarifiedRoute" }
                ];
            }

            // и кнопка “искать в базе”
            buttons.push({ text: "🔎 Поиск по базе знаний", transition: "/RAGHandler/Query" });
            buttons.push({ text: "Задать другой вопрос", transition: "/ClassifierRouter/Route" });

            $reactions.buttons(buttons);


    state: ClarifiedRoute
        script:
            // Определим категорию по тексту кнопки
            var t = ($request.query || "").toLowerCase();

            if (t.indexOf("статус") >= 0) $session.category = "order_status";
            else if (t.indexOf("оплат") >= 0) $session.category = "payment_refund";
            else if (t.indexOf("возврат") >= 0 || t.indexOf("обмен") >= 0) $session.category = "return_exchange";
            else if (t.indexOf("достав") >= 0) $session.category = "delivery";
            else if (t.indexOf("товар") >= 0) $session.category = "product_info";
            else if (t.indexOf("общ") >= 0) $session.category = "general_info";
            else if (t.indexOf("промо") >= 0 || t.indexOf("бонус") >= 0) $session.category = "promo_loyalty";
            else if (t.indexOf("аккаунт") >= 0 || t.indexOf("личн") >= 0) $session.category = "account";
            else if (t.indexOf("тех") >= 0) $session.category = "technical_issue";
            else $session.category = "other";

            $reactions.transition("/ClassifierRouter/RouteByCategory");


    state: FallbackMenu
        a: Я могу помочь с типовыми вопросами или поискать ответ в базе знаний. Что выберем?
        buttons:
            "❓ Частые вопросы" -> /ClassifierRouter/Route
            "🔎 Поиск по базе знаний" -> /RAGHandler/Query
            "📦 Статус заказа" -> /OrderStatus/AskOrderNumber



theme: /RAGHandler

    state: Query
        script:
            var q = ($session.ragQuery || $session.lastUserText || $request.query || "").trim();

            if (!q) {
                $reactions.answer("Напишите вопрос текстом, и я попробую найти ответ в базе знаний.");
                $reactions.transition("/ClassifierRouter/Route");
                return;
            }

            var apiBase = $global.API_BASE_URL;
            var apiKey  = $global.API_KEY;

            try {
                var ragResp = $http.query(apiBase + "/api/v1/rag/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": apiKey
                },
                body: { query: q, top_k: 3, session_id: $session.id },
                timeout: 4000
                });

                var ragStatus = ragResp.status || ragResp.statusCode;
                var ragData = ragResp.data || ragResp.body || {};

                if (ragStatus != 200) {
                log("RAG bad status=" + ragStatus);
                $reactions.transition("/RAGHandler/NoAnswer");
                return;
                }

                if (ragData.is_confident && ragData.answer) {
                $session.lastSource = "rag";
                $session.failCount = 0;

                $reactions.answer(ragData.answer);

                // (опционально) покажем источники одной кнопкой, но без перегруза
                $reactions.buttons([
                    { text: "Задать другой вопрос", transition: "/ClassifierRouter/Route" },
                    { text: "📦 Статус заказа", transition: "/OrderStatus/AskOrderNumber" }
                ]);
                return;
                }

                // если не уверен — честный отказ
                $reactions.transition("/RAGHandler/NoAnswer");
                return;

            } catch (e) {
                log("RAG error: " + e.message);
                $reactions.transition("/RAGHandler/NoAnswer");
                return;
            }


    state: NoAnswer
        a: К сожалению, я не нашёл точного ответа в базе знаний. Попробуйте переформулировать вопрос или выберите сценарий:
        buttons:
            "Задать другой вопрос" -> /ClassifierRouter/Route
            "📦 Статус заказа" -> /OrderStatus/AskOrderNumber