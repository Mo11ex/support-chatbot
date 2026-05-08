import re
import time
from app.config import settings


def normalize_text(text: str) -> str:
    text = str(text).replace("\xa0", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_order_number(text: str) -> str | None:
    pattern = re.compile(r"\b([A-Z0-9]{6,12}|\d{5,12})\b")
    matches = pattern.findall(text.upper())
    return matches[0] if matches else None


def is_order_tracking_query(text: str) -> bool:
    """
    Отличаем запросы про статус/трекинг заказа
    от запросов про изменение/отмену/процесс заказа.
    """
    t = normalize_text(text).lower()

    process_markers = [
        "отмен",
        "измен",
        "помен",
        "добав",
        "убра",
        "адрес",
        "оплат",
        "состав заказа",
        "доставк",
        "перенест",
        "ускор",
    ]

    tracking_markers = [
        "где мой заказ",
        "статус заказа",
        "где заказ",
        "мой заказ",
        "заказ где",
        "посылка",
        "трек",
        "трекинг",
        "отслед",
        "отслеж",
        "номер заказа",
    ]

    if any(marker in t for marker in process_markers):
        return False

    if extract_order_number(t):
        return True

    return any(marker in t for marker in tracking_markers)


class RouterService:
    def __init__(self, classifier, faq_service, rag_service, logger, llm_service):
        self.classifier = classifier
        self.faq = faq_service
        self.rag = rag_service
        self.logger = logger
        self.llm = llm_service

        self.faq_upper = settings.faq_upper_threshold
        self.faq_lower = settings.faq_lower_threshold
        self.rag_upper = settings.rag_upper_threshold
        self.rag_lower = settings.rag_lower_threshold
        self.conf_high = settings.classifier_confidence_high
        self.conf_mid = settings.classifier_confidence_mid

        self.category_map = settings.classifier_to_rag_category
        self.faq_primary = set(settings.faq_primary_intents)
        self.rag_primary = set(settings.rag_primary_intents)
        self.fallback_intents = set(settings.fallback_intents)
        
    def _render_faq_answer(self, query: str, faq_result: dict) -> str:
        """
        Для FAQ НЕ используем LLM.
        FAQ already curated and should stay fast.
        """
        return faq_result.get("top1_text") or ""

    def _render_rag_answer(self, query: str, rag_result: dict) -> str:
        """
        Генерация ответа по RAG-контексту. Склеиваем top-3 чанка.
        """
        if "results" in rag_result and rag_result["results"]:
            chunks = [res.get("text", "") for res in rag_result["results"][:3] if res.get("text")]
            context = "\n---\n".join(chunks)
        else:
            context = rag_result.get("top1_text") or ""

        if self.llm:
            return self.llm.generate_answer(query, context)
        return context

    async def handle(self, text: str, order_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        norm_text = normalize_text(text)

        faq_score = None
        rag_score = None
        intent = None
        intent_confidence = None
        branch = None
        answer = None
        source_type = None
        source_id = None
        fallback_recommended = False

        # ── Шаг 1: Classifier first ────────────────────────────────────────
        clf_result = self.classifier.predict(norm_text)
        intent = clf_result["category"]
        intent_confidence = clf_result["confidence"]

        # ── Шаг 2: Other / fallback early ─────────────────────────────────
        if intent in self.fallback_intents:
            branch = "fallback"
            answer = "Не совсем понял вопрос. Попробуйте переформулировать его или напишите оператору."
            source_type = "system"
            fallback_recommended = True

            latency = (time.perf_counter() - t0) * 1000
            self.logger.log(
                text=norm_text,
                branch=branch,
                intent=intent,
                intent_confidence=intent_confidence,
                faq_score=None,
                rag_score=None,
                source_type=source_type,
                source_id=None,
                answer=answer,
                latency_ms=latency,
                fallback_recommended=True,
            )
            return self._build_response(
                answer=answer,
                branch=branch,
                intent=intent,
                confidence=intent_confidence,
                faq_score=None,
                rag_score=None,
                source_type=source_type,
                source_id=None,
                latency_ms=latency,
                fallback_recommended=True,
            )

        # ── Шаг 3: Order status split ──────────────────────────────────────
        if intent == "order_status":
            if is_order_tracking_query(norm_text):
                order_number = order_id or extract_order_number(norm_text)
                if not order_number:
                    branch = "need_order_id"
                    answer = "Назовите номер заказа, и я проверю его статус."
                    source_type = "system"
                    source_id = None

                    latency = (time.perf_counter() - t0) * 1000
                    self.logger.log(
                        text=norm_text,
                        branch=branch,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        faq_score=None,
                        rag_score=None,
                        source_type=source_type,
                        source_id=source_id,
                        answer=answer,
                        latency_ms=latency,
                    )
                    return self._build_response(
                        answer=answer,
                        branch=branch,
                        intent=intent,
                        confidence=intent_confidence,
                        faq_score=None,
                        rag_score=None,
                        source_type=source_type,
                        source_id=source_id,
                        latency_ms=latency,
                    )
                else:
                    branch = "orders_api"
                    source_type = "orders"
                    source_id = order_number
                    answer = None

                    latency = (time.perf_counter() - t0) * 1000
                    # ВАЖНО: здесь НЕ логируем, т.к. реальный ответ будет построен в answer.py
                    return self._build_response(
                        answer=answer,
                        branch=branch,
                        intent=intent,
                        confidence=intent_confidence,
                        faq_score=None,
                        rag_score=None,
                        source_type=source_type,
                        source_id=source_id,
                        latency_ms=latency,
                    )
            else:
                # Это не трекинг, а knowledge/process вопрос про заказ
                rag_category = self.category_map.get("order_status")
                rag_result = self.rag.search(norm_text, category=rag_category)
                rag_score = rag_result["top1_score"]

                if rag_score >= self.rag_upper:
                    answer = self._render_rag_answer(norm_text, rag_result)
                    branch = "rag_with_filter"
                    source_type = "rag"
                    source_id = rag_result["top1_source_file"]

                    latency = (time.perf_counter() - t0) * 1000
                    self.logger.log(
                        text=norm_text,
                        branch=branch,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        faq_score=None,
                        rag_score=rag_score,
                        source_type=source_type,
                        source_id=source_id,
                        answer=answer,
                        latency_ms=latency,
                    )
                    return self._build_response(
                        answer=answer,
                        branch=branch,
                        intent=intent,
                        confidence=intent_confidence,
                        faq_score=None,
                        rag_score=rag_score,
                        source_type=source_type,
                        source_id=source_id,
                        latency_ms=latency,
                    )

                branch = "fallback"
                answer = "Не нашёл точного ответа по заказу. Попробуйте уточнить вопрос или обратитесь к оператору."
                source_type = "system"
                fallback_recommended = True

                latency = (time.perf_counter() - t0) * 1000
                self.logger.log(
                    text=norm_text,
                    branch=branch,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    faq_score=None,
                    rag_score=rag_score,
                    source_type=source_type,
                    source_id=None,
                    answer=answer,
                    latency_ms=latency,
                    fallback_recommended=True,
                )
                return self._build_response(
                    answer=answer,
                    branch=branch,
                    intent=intent,
                    confidence=intent_confidence,
                    faq_score=None,
                    rag_score=rag_score,
                    source_type=source_type,
                    source_id=None,
                    latency_ms=latency,
                    fallback_recommended=True,
                )

        # ── Шаг 4: FAQ retrieval for all non-fallback/non-order queries ────
        faq_result = self.faq.search(norm_text)
        faq_score = faq_result["top1_score"]

        # FAQ primary intents always may use FAQ
        if intent in self.faq_primary and faq_score >= self.faq_upper:
            answer = self._render_faq_answer(norm_text, faq_result)
            branch = "faq_direct"
            source_type = "faq"
            source_id = faq_result["top1_doc_id"]

            latency = (time.perf_counter() - t0) * 1000
            self.logger.log(
                text=norm_text,
                branch=branch,
                intent=intent,
                intent_confidence=intent_confidence,
                faq_score=faq_score,
                rag_score=None,
                source_type=source_type,
                source_id=source_id,
                answer=answer,
                latency_ms=latency,
            )
            return self._build_response(
                answer=answer,
                branch=branch,
                intent=intent,
                confidence=intent_confidence,
                faq_score=faq_score,
                rag_score=None,
                source_type=source_type,
                source_id=source_id,
                latency_ms=latency,
            )

        # Если classifier не слишком уверен, но FAQ очень уверен — доверяем FAQ
        if intent_confidence < self.conf_high and faq_score >= self.faq_upper:
            answer = self._render_faq_answer(norm_text, faq_result)
            branch = "faq_direct"
            source_type = "faq"
            source_id = faq_result["top1_doc_id"]

            latency = (time.perf_counter() - t0) * 1000
            self.logger.log(
                text=norm_text,
                branch=branch,
                intent=intent,
                intent_confidence=intent_confidence,
                faq_score=faq_score,
                rag_score=None,
                source_type=source_type,
                source_id=source_id,
                answer=answer,
                latency_ms=latency,
            )
            return self._build_response(
                answer=answer,
                branch=branch,
                intent=intent,
                confidence=intent_confidence,
                faq_score=faq_score,
                rag_score=None,
                source_type=source_type,
                source_id=source_id,
                latency_ms=latency,
            )

        # ── Шаг 5: RAG primary / fallback from FAQ ─────────────────────────
        rag_category = self.category_map.get(intent)
        if rag_category is not None and intent_confidence >= self.conf_mid:
            rag_result = self.rag.search(norm_text, category=rag_category)
            rag_score = rag_result["top1_score"]

            if rag_score >= self.rag_upper:
                answer = self._render_rag_answer(norm_text, rag_result)
                branch = "rag_with_filter"
                source_type = "rag"
                source_id = rag_result["top1_source_file"]

                latency = (time.perf_counter() - t0) * 1000
                self.logger.log(
                    text=norm_text,
                    branch=branch,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    faq_score=faq_score,
                    rag_score=rag_score,
                    source_type=source_type,
                    source_id=source_id,
                    answer=answer,
                    latency_ms=latency,
                )
                return self._build_response(
                    answer=answer,
                    branch=branch,
                    intent=intent,
                    confidence=intent_confidence,
                    faq_score=faq_score,
                    rag_score=rag_score,
                    source_type=source_type,
                    source_id=source_id,
                    latency_ms=latency,
                )

        # ── Шаг 6: RAG no-filter for medium-confidence cases ───────────────
        if intent_confidence >= self.conf_mid:
            rag_result_nf = self.rag.search(norm_text, category=None)
            rag_score = rag_result_nf["top1_score"]

            if rag_score >= self.rag_upper:
                answer = self._render_rag_answer(norm_text, rag_result_nf)
                branch = "rag_no_filter"
                source_type = "rag"
                source_id = rag_result_nf["top1_source_file"]

                latency = (time.perf_counter() - t0) * 1000
                self.logger.log(
                    text=norm_text,
                    branch=branch,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    faq_score=faq_score,
                    rag_score=rag_score,
                    source_type=source_type,
                    source_id=source_id,
                    answer=answer,
                    latency_ms=latency,
                )
                return self._build_response(
                    answer=answer,
                    branch=branch,
                    intent=intent,
                    confidence=intent_confidence,
                    faq_score=faq_score,
                    rag_score=rag_score,
                    source_type=source_type,
                    source_id=source_id,
                    latency_ms=latency,
                )

        # ── Шаг 7: Final fallback ───────────────────────────────────────────
        branch = "fallback"
        fallback_recommended = True
        source_type = "system"

        if intent_confidence < self.conf_mid:
            answer = "Не уверен, что правильно понял вопрос. Попробуйте переформулировать его или обратиться к оператору."
        else:
            answer = "Не нашёл ответа в базе знаний. Попробуйте уточнить вопрос или обратитесь к оператору."

        latency = (time.perf_counter() - t0) * 1000
        self.logger.log(
            text=norm_text,
            branch=branch,
            intent=intent,
            intent_confidence=intent_confidence,
            faq_score=faq_score,
            rag_score=rag_score,
            source_type=source_type,
            source_id=None,
            answer=answer,
            latency_ms=latency,
            fallback_recommended=True,
        )
        return self._build_response(
            answer=answer,
            branch=branch,
            intent=intent,
            confidence=intent_confidence,
            faq_score=faq_score,
            rag_score=rag_score,
            source_type=source_type,
            source_id=None,
            latency_ms=latency,
            fallback_recommended=True,
        )

    def _build_response(
        self,
        answer: str | None,
        branch: str,
        intent: str | None,
        confidence: float | None,
        faq_score: float | None,
        rag_score: float | None,
        source_type: str | None,
        source_id: str | None,
        latency_ms: float,
        fallback_recommended: bool = False,
    ) -> dict:
        return {
            "answer": answer,
            "branch": branch,
            "intent": intent,
            "intent_confidence": confidence,
            "source_type": source_type,
            "source_id": source_id,
            "faq_score": faq_score,
            "rag_score": rag_score,
            "fallback_recommended": fallback_recommended,
            "latency_ms": round(latency_ms, 2),
        }