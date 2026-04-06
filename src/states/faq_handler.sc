theme: /FAQHandler

    state: ShowMenu
        a: Популярные вопросы:
        buttons:
            "🔄 Как вернуть товар?" -> /FAQHandler/Return
            "💳 Способы оплаты" -> /FAQHandler/Payment
            "🚚 Доставка" -> /FAQHandler/Delivery
            "🏠 Главное меню" -> /Greeting/Help
    
    state: Return
        a: Для возврата товара:
        a: 1. Перейдите в «Мои заказы»
        a: 2. Выберите заказ → «Оформить возврат»
        a: 3. Заполните форму
        a: 
        a: Возврат возможен в течение 14 дней. Товар должен сохранять товарный вид.
        buttons:
            "✅ Понятно" -> /Greeting/Help
            "❓ Другой вопрос" -> /FAQHandler/ShowMenu
    
    state: Payment
        a: Способы оплаты:
        a: • Банковская карта (Visa, MC, МИР)
        a: • СБП
        a: • Apple Pay / Google Pay
        a: • Наложенный платёж (+50₽)
        buttons:
            "✅ Понятно" -> /Greeting/Help
    
    state: Delivery
        a: Доставка:
        a: • Москва и СПб: от 300₽, бесплатно от 5000₽
        a: • Регионы: от 400₽, бесплатно от 7000₽
        a: • Самовывоз СДЭК: бесплатно
        a: 
        a: Срок: Москва 1-2 дня, регионы 3-7 дней.
        buttons:
            "✅ Понятно" -> /Greeting/Helptheme: /FAQHandler

    state: ShowMenu
        a: Популярные вопросы:
        buttons:
            "🔄 Как вернуть товар?" -> /FAQHandler/Return
            "💳 Способы оплаты" -> /FAQHandler/Payment
            "🚚 Доставка" -> /FAQHandler/Delivery
            "🏠 Главное меню" -> /Greeting/Help
    
    state: Return
        a: Для возврата товара:
        a: 1. Перейдите в «Мои заказы»
        a: 2. Выберите заказ → «Оформить возврат»
        a: 3. Заполните форму
        a: 
        a: Возврат возможен в течение 14 дней. Товар должен сохранять товарный вид.
        buttons:
            "✅ Понятно" -> /Greeting/Help
            "❓ Другой вопрос" -> /FAQHandler/ShowMenu
    
    state: Payment
        a: Способы оплаты:
        a: • Банковская карта (Visa, MC, МИР)
        a: • СБП
        a: • Apple Pay / Google Pay
        a: • Наложенный платёж (+50₽)
        buttons:
            "✅ Понятно" -> /Greeting/Help
    
    state: Delivery
        a: Доставка:
        a: • Москва и СПб: от 300₽, бесплатно от 5000₽
        a: • Регионы: от 400₽, бесплатно от 7000₽
        a: • Самовывоз СДЭК: бесплатно
        a: 
        a: Срок: Москва 1-2 дня, регионы 3-7 дней.
        buttons:
            "✅ Понятно" -> /Greeting/Help