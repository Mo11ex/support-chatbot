theme: /FAQHandler

    state: ShowMenu
        #TODO: intent:
        a: Популярные вопросы:
        buttons:
            "🔄 Как вернуть товар?" -> /FAQHandler/Return
            "💳 Способы оплаты" -> /FAQHandler/Payment
            "🚚 Доставка" -> /FAQHandler/Delivery
            "🏠 Главное меню" -> /Greeting/Help
    
    state: Return
        #TODO: intent:
        a: Для возврата товара:
         1. Перейдите в «Мои заказы»
         2. Выберите заказ → «Оформить возврат»
         3. Заполните форму
        a:Возврат возможен в течение 14 дней. Товар должен сохранять товарный вид.
        
        buttons:
            "✅ Понятно" -> /Greeting/Help
            "❓ Другой вопрос" -> /FAQHandler/ShowMenu
    
    state: Payment
        #TODO: intent:
        a: Способы оплаты:
         • Банковская карта (Visa, MC, МИР)
         • СБП
         • Apple Pay / Google Pay
         • Наложенный платёж (+50₽)
         
        buttons:
            "✅ Понятно" -> /Greeting/Help
            "❓ Другой вопрос" -> /FAQHandler/ShowMenu
    
    state: Delivery
        #TODO: intent:
        a: Доставка:
         • Москва и СПб: от 300₽, бесплатно от 5000₽
         • Регионы: от 400₽, бесплатно от 7000₽
         • Самовывоз СДЭК: бесплатно
        a: Срок: Москва 1-2 дня, регионы 3-7 дней.
        
        buttons:
            "✅ Понятно" -> /Greeting/Help
            "❓ Другой вопрос" -> /FAQHandler/ShowMenu