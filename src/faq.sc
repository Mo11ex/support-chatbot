# ══════════════════════════════════════════════
# FAQ — быстрые ответы
# ══════════════════════════════════════════════

theme: /FAQ

    # ── Прямые FAQ (по кнопкам) ──

    state: ReturnInfo
        q!: как вернуть товар
        q!: хочу вернуть товар
        q!: возврат товара
        q!: как оформить возврат
        q!: вернуть покупку
        q!: условия возврата
        a: Для возврата товара:
        a: 1️⃣ Перейдите в «Мои заказы» в личном кабинете
        a: 2️⃣ Выберите заказ и нажмите «Оформить возврат»
        a: 3️⃣ Заполните форму и выберите причину
        a: ⏰ Срок возврата: 14 дней. Товар должен сохранять товарный вид.
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "📦 Статус заказа" -> /OrderFlow/AskOrderNumber
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: DeliveryInfo
        q!: сколько стоит доставка
        q!: стоимость доставки
        q!: способы доставки
        q!: как доставляете
        q!: есть самовывоз
        q!: бесплатная доставка
        q!: как оформить доставку
        q!: сроки доставки
        a: 🚚 Доставка:
        a: • Москва и СПб: от 300₽, бесплатно от 5000₽
        a: • Регионы: от 400₽, бесплатно от 7000₽
        a: • Самовывоз из пунктов СДЭК: бесплатно
        a: ⏰ Сроки: Москва 1-2 дня, регионы 3-7 рабочих дней.
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: PaymentInfo
        q!: как оплатить
        q!: способы оплаты
        q!: оплата картой
        q!: оплата при получении
        a: 💳 Способы оплаты:
        a: • Банковская карта (Visa, MasterCard, МИР)
        a: • СБП (Система быстрых платежей)
        a: • Apple Pay / Google Pay
        a: • Наложенный платёж при получении (+50₽)
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: RefundInfo
        q!: когда вернут деньги
        q!: возврат денег
        q!: верните деньги
        q!: статус возврата денег
        a: 💰 Возврат денежных средств осуществляется в течение 10 рабочих дней на карту, с которой была произведена оплата.
        a: Статус возврата можно отслеживать в личном кабинете в разделе «Мои возвраты».
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: PromoInfo
        q!: промокод
        q!: как ввести промокод
        q!: промокод не работает
        q!: купон на скидку
        q!: скидки
        q!: какие акции
        q!: есть промокоды
        q!: дай промокод
        a: 🏷️ Для применения промокода:
        a: 1️⃣ Добавьте товары в корзину
        a: 2️⃣ На странице оформления найдите поле «Промокод»
        a: 3️⃣ Введите код и нажмите «Применить»
        a: Если промокод не работает — проверьте срок действия и условия акции.
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: PasswordInfo
        q!: забыл пароль
        q!: не могу войти
        q!: сбросить пароль
        q!: восстановить пароль
        a: 🔐 Для восстановления пароля:
        a: 1️⃣ На странице входа нажмите «Забыли пароль?»
        a: 2️⃣ Введите email при регистрации
        a: 3️⃣ Проверьте почту — придёт ссылка для сброса
        a: 4️⃣ Перейдите по ссылке и задайте новый пароль
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "👨‍💼 Оператор" -> /Escalation/Connect

    state: ContactsInfo
        q!: контакты
        q!: телефон
        q!: как связаться
        q!: номер телефона
        q!: email поддержки
        q!: ваш телефон
        q!: режим работы
        q!: часы работы
        q!: во сколько работаете
        a: 📞 Контакты:
        a: ☎️ 8-800-123-45-67 (бесплатно по России)
        a: 📧 support@example-shop.ru
        a: 💬 Чат на сайте (вы уже здесь!)
        a: 🕐 Поддержка: 9:00–21:00 МСК, без выходных
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "📦 Статус заказа" -> /OrderFlow/AskOrderNumber

    state: GaranteeInfo
        q!: гарантия
        q!: гарантийный срок
        q!: гарантия на товар
        q!: товар сломался
        a: 🛡️ Гарантийные сроки:
        a: • Электроника: 12 месяцев
        a: • Одежда и обувь: 30 дней
        a: • Аксессуары: 14 дней
        a: Для гарантийного обслуживания обратитесь в поддержку с фото/видео дефекта.
        buttons:
            "👍 Помогло" -> /Feedback/Positive
            "👎 Не помогло" -> /Feedback/Negative
            "👨‍💼 Оператор" -> /Escalation/Connect

    # ── FAQ через API (для нераспознанных) ──

    state: SearchFAQ
        script:
            var query = $request.query;
            try {
                var resp = $http.post($session.API_URL + "/api/v1/faq/match", {
                    headers: {"Content-Type": "application/json"},
                    body: {"text": query, "session_id": $request.channelUserId},
                    timeout: 5000
                });

                if (resp.status == 200 && resp.body.matched) {
                    $session.faqMatched = true;
                    $session.faqAnswer = resp.body.answer;
                    $session.faqCategory = resp.body.category;
                } else {
                    $session.faqMatched = false;
                }
            } catch(e) {
                $session.faqMatched = false;
            }

        if: $session.faqMatched
            a: {{$session.faqAnswer}}
            buttons:
                "👍 Помогло" -> /Feedback/Positive
                "👎 Не помогло" -> /Feedback/Negative
                "👨‍💼 Оператор" -> /Escalation/Connect
        else:
            go!: /Router/Classify