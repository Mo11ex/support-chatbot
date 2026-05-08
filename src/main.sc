require: slotfilling/slotFilling.sc
  module = sys.zb-common

require: functions.js

theme: /

    # ─────────────────────────────────────────────────────────────
    # START
    # ─────────────────────────────────────────────────────────────

    state: Start
        q!: $regex</start>
        a: Здравствуйте! Я умный помощник службы поддержки.
        a: Могу помочь с доставкой, оплатой, возвратами и статусом заказа.
        script:
            $session.awaitingOrder = false;
            $session.lastQuery = null;

            $reactions.buttons([
                "Где мой заказ?",
                "Как вернуть товар?",
                "Условия доставки",
                "Способы оплаты"
            ]);

    # ─────────────────────────────────────────────────────────────
    # HELP
    # ─────────────────────────────────────────────────────────────

    state: Help
        q!: помощь
        q!: $regex</help>
        a: Я могу:
        a: — подсказать статус заказа;
        a: — ответить на вопросы по доставке, оплате и возврату;
        a: — помочь с аккаунтом и техническими проблемами.
        a: Просто напишите вопрос свободным текстом.
        script:
            $reactions.buttons([
                "Где мой заказ?",
                "Как вернуть товар?",
                "Позови оператора"
            ]);

    # ─────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────

    state: Reset
        q!: начать заново
        q!: сначала
        q!: reset
        q!: сброс
        script:
            $session.awaitingOrder = false;
            $session.lastQuery = null;
            $reactions.answer("Хорошо, начнём заново.");
            $reactions.transition("/Start");

    # ─────────────────────────────────────────────────────────────
    # OPERATOR
    # ─────────────────────────────────────────────────────────────

    state: Operator
        q!: оператор
        q!: позови оператора
        q!: нужен оператор
        q!: хочу поговорить с человеком
        q!: $regex</operator>
        script:
            $session.awaitingOrder = false;
            $reactions.answer("Передаю вопрос оператору. Если оператор сейчас недоступен, попробуйте повторить запрос позже.");
            $reactions.buttons([
                "Начать заново"
            ]);

    # ─────────────────────────────────────────────────────────────
    # FEEDBACK
    # ─────────────────────────────────────────────────────────────

    state: FeedbackPositive
        q!: 👍 Помогло
        script:
            $session.awaitingOrder = false;
            $reactions.answer("Отлично! Рад был помочь 😊");
            $reactions.buttons([
                "Ещё вопрос",
                "Начать заново"
            ]);

    state: FeedbackNegative
        q!: 👎 Не помогло
        script:
            $reactions.answer("Жаль, что не помог. Могу предложить оператора или попробовать ещё раз.");
            $reactions.buttons([
                "Позови оператора",
                "Начать заново"
            ]);

    state: AnotherQuestion
        q!: ещё вопрос
        script:
            $session.awaitingOrder = false;
            $session.lastQuery = null;
            $reactions.answer("Хорошо, задайте следующий вопрос.");

    # ─────────────────────────────────────────────────────────────
    # MAIN CATCH ALL
    # ─────────────────────────────────────────────────────────────

    state: CatchAll
        q!: *
        script:
            // 1. Если ждём номер заказа — обрабатываем это как order_id
            if ($session.awaitingOrder === true) {
                if ($request.query === "Отмена" || $request.query === "начать заново" || $request.query === "сначала") {
                    $session.awaitingOrder = false;
                    $session.lastQuery = null;
                    $reactions.answer("Хорошо, отменяю запрос номера заказа.");
                    $reactions.buttons([
                        "Начать заново"
                    ]);
                    return;
                }

                var resOrder = askBackend($session.lastQuery || "где мой заказ?", $request.query, $request.channelUserId);

                if (resOrder.isOk && resOrder.data) {
                    var dataOrder = resOrder.data;

                    if (dataOrder.answer) {
                        $reactions.answer(dataOrder.answer);
                    } else {
                        $reactions.answer("Не удалось получить информацию по заказу.");
                    }

                    $session.awaitingOrder = false;
                    $session.lastQuery = null;

                    if (dataOrder.fallback_recommended === true) {
                        $reactions.buttons([
                            "Позови оператора",
                            "Начать заново"
                        ]);
                    } else {
                        $reactions.buttons([
                            "👍 Помогло",
                            "👎 Не помогло"
                        ]);
                    }

                } else {
                    $session.awaitingOrder = false;
                    $reactions.answer("Извините, сервис проверки заказов сейчас недоступен.");
                    $reactions.buttons([
                        "Позови оператора",
                        "Начать заново"
                    ]);
                }

                return;
            }

            // 2. Обычный запрос
            $session.lastQuery = $request.query;

            var res = askBackend($request.query, null, $request.channelUserId);

            if (res.isOk && res.data) {
                var data = res.data;

                if (data.answer) {
                    $reactions.answer(cleanAnswer(data.answer));
                } else {
                    $reactions.answer("Не удалось сформировать ответ.");
                }

                if (data.branch === "need_order_id") {
                    $session.awaitingOrder = true;
                    $session.lastQuery = $request.query;

                    $reactions.buttons([
                        "Отмена"
                    ]);
                    return;
                }

                if (data.fallback_recommended === true) {
                    $reactions.buttons([
                        "Позови оператора",
                        "Начать заново"
                    ]);
                } else {
                    $reactions.buttons([
                        "👍 Помогло",
                        "👎 Не помогло"
                    ]);
                }

            } else {
                $reactions.answer("Извините, технический сбой на сервере. Попробуйте повторить запрос позже.");
                $reactions.buttons([
                    "Начать заново"
                ]);
            }