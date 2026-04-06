from __future__ import annotations

from textwrap import dedent

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ecopulse.config import AppConfig
from ecopulse.knowledge_base import KnowledgeBase

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except Exception:
    ChatOpenAI = None
    OPENAI_AVAILABLE = False


class EcoPulseRAG:
    def __init__(self, config: AppConfig, knowledge_base: KnowledgeBase) -> None:
        self.config = config
        self.knowledge_base = knowledge_base
        self.llm = self._build_llm()
        self.prompt = ChatPromptTemplate.from_template(
            dedent(
                """
                You are Eco-Pulse, an environmental decision assistant for smart cities.
                Use the live city snapshot and retrieved guidance to answer the user's question.
                Keep the answer practical, evidence-based, and easy to act on.

                City: {city}
                User question: {question}

                Live snapshot:
                {snapshot}

                Retrieved context:
                {context}

                Respond with:
                1. A short direct answer.
                2. Recommended action with timing guidance if possible.
                3. A brief note citing which environmental factors drive the recommendation.
                """
            ).strip()
        )

    def answer_question(self, city: str, snapshot: dict, question: str) -> dict:
        search_text = self._build_search_text(city, snapshot, question)
        documents = self.knowledge_base.retrieve(search_text)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in documents
        )

        if self.llm is not None:
            chain = self.prompt | self.llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "city": city,
                    "question": question,
                    "snapshot": self._snapshot_text(snapshot),
                    "context": context,
                }
            )
        else:
            answer = self._fallback_answer(city, snapshot, question, documents)

        return {
            "answer": answer,
            "evidence": [
                {"source": doc.metadata.get("source", "unknown"), "content": doc.page_content}
                for doc in documents
            ],
        }

    def _build_llm(self):
        if not self.config.openai_api_key or not OPENAI_AVAILABLE:
            return None
        return ChatOpenAI(
            model=self.config.openai_model,
            api_key=self.config.openai_api_key,
            temperature=0.2,
        )

    @staticmethod
    def _build_search_text(city: str, snapshot: dict, question: str) -> str:
        return (
            f"{city} environmental guidance {question} "
            f"PM2.5 {snapshot['pm2_5']} UV {snapshot['uv_index']} "
            f"humidity {snapshot['relative_humidity_2m']} wind {snapshot['wind_speed_10m']}"
        )

    @staticmethod
    def _snapshot_text(snapshot: dict) -> str:
        return (
            f"AQI category: {snapshot['aqi_category']}; PM2.5: {snapshot['pm2_5']:.1f} ug/m3; "
            f"temperature: {snapshot['temperature_2m']:.1f} C; humidity: {snapshot['relative_humidity_2m']:.0f}%; "
            f"wind: {snapshot['wind_speed_10m']:.1f} km/h; UV index: {snapshot['uv_index']:.1f}; "
            f"exposure score: {snapshot.get('exposure_score', 0):.1f}."
        )

    def _fallback_answer(self, city: str, snapshot: dict, question: str, documents: list) -> str:
        pm = float(snapshot["pm2_5"])
        uv = float(snapshot["uv_index"])
        wind = float(snapshot["wind_speed_10m"])
        humidity = float(snapshot["relative_humidity_2m"])

        if pm <= 12 and uv < 6:
            timing = "Conditions are favorable now and during the next low-traffic period."
            action = "Outdoor activity is reasonable, though morning and late evening remain the safest windows."
        elif pm <= 35.4:
            timing = "Prefer early morning or later evening when traffic-related exposure is typically lower."
            action = "Limit prolonged roadside exposure and choose parks or low-congestion streets."
        else:
            timing = "Delay outdoor exertion until pollution drops, ideally after a weather or traffic shift."
            action = "Short essential trips are fine with a mask, but strenuous outdoor activity should be postponed."

        modifiers = []
        if uv >= 7:
            modifiers.append("UV is elevated, so midday outdoor time should be minimized")
        if humidity >= 75:
            modifiers.append("high humidity may make exertion feel harder")
        if wind >= 12:
            modifiers.append("wind may help disperse some local pollutants")

        sources = ", ".join(doc.metadata.get("source", "unknown") for doc in documents[:3]) or "the indexed guidance"
        modifier_text = "; ".join(modifiers) if modifiers else "current weather is otherwise manageable"

        return (
            f"For {city}, the current air-quality category is {snapshot['aqi_category']}. {timing} {action} "
            f"This recommendation is driven mainly by PM2.5 at {pm:.1f} ug/m3, UV at {uv:.1f}, humidity at "
            f"{humidity:.0f}%, and wind speed at {wind:.1f} km/h. Additional context: {modifier_text}. "
            f"Relevant guidance was retrieved from {sources}. Question considered: {question}"
        )
