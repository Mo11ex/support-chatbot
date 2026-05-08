var BACKEND_URL = "https://74c8018b-d87c-4adc-ac88-071708caed89.tunnel4.com/api/v1/answer";

function askBackend(text, orderId, userId) {
    var body = {
        "text": text,
        "session_id": userId || "default_user"
    };
    
    if (orderId) {
        body["order_id"] = orderId;
    }
    
    var response = $http.post(BACKEND_URL, {
        headers: {
            "Content-Type": "application/json"
        },
        body: body,
        timeout: 25000
    });
    
    return response;
}

function cleanAnswer(text) {
    if (!text) return "";

    var result = text;

    // убрать markdown-разметку заголовков
    result = result.replace(/^###\s+/gm, "");
    result = result.replace(/^##\s+/gm, "");
    result = result.replace(/^#\s+/gm, "");

    // убрать горизонтальные линии
    result = result.replace(/---+/g, "");

    // убрать лишние пустые строки
    result = result.replace(/\n{3,}/g, "\n\n");

    // trim
    result = result.trim();

    return result;
}