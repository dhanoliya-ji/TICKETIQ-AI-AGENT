"""One ticket end to end: classify, score sentiment and urgency, let the bandit pick a configuration,
retrieve policy passages, run the ReAct agent; then turn customer feedback into a bandit update."""
from __future__ import annotations

import time

from .agent import LlmClient, TriageAgent
from .bandit import EpsilonGreedyContextualBandit, all_config_names, build_state_key, compute_reward, get_config
from .classifier import TicketClassifierService
from .config import SETTINGS
from .retrieval import KnowledgeRetriever
from .sentiment import AspectSentimentAnalyzer, score_urgency, urgency_bucket


class TicketIQ:
    def __init__(self) -> None:
        self.classifier = TicketClassifierService(SETTINGS.tickets_file, SETTINGS.test_split, SETTINGS.random_seed).train()
        self.sentiment = AspectSentimentAnalyzer()
        self.retriever = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load()
        self.agent = TriageAgent(LlmClient())
        self.bandit = EpsilonGreedyContextualBandit(all_config_names(), SETTINGS.bandit_epsilon, SETTINGS.random_seed)

    def handle(self, subject: str, body: str, customer_tier: str = "pro") -> dict:
        started = time.perf_counter()
        classification = self.classifier.classify(subject, body)
        aspects = self.sentiment.analyse(subject, body)
        urgency = score_urgency(classification.category, aspects, customer_tier)
        bucket = urgency_bucket(urgency)
        state_key = build_state_key(classification.category, bucket, customer_tier)
        config_name, reason = self.bandit.select_action(state_key)
        config = get_config(config_name)
        hits = self.retriever.retrieve(subject, body, classification.category, config.rag_top_k)
        snippets = [h.as_dict() for h in hits]
        result = self.agent.run(subject=subject, body=body, customer_tier=customer_tier,
                                category=classification.category, urgency_bucket=bucket,
                                snippets=snippets, config=config)
        return {"category": classification.category, "category_confidence": classification.confidence,
                "aspect_sentiments": [a.as_dict() for a in aspects], "urgency_score": round(urgency, 4),
                "urgency_bucket": bucket, "state_key": state_key, "config": config_name,
                "config_selection_reason": reason, "retrieved_knowledge": snippets, "action": result.decision,
                "response_text": result.response_text, "reasoning_trace": result.as_dict()["reasoning_trace"],
                "llm_backend": result.llm_backend, "latency_seconds": round(time.perf_counter() - started, 4)}

    def feedback(self, ticket: dict, helpful: int) -> float:
        """helpful = 1 or 0. The reward trades helpfulness against latency."""
        reward = compute_reward(helpful, ticket["latency_seconds"])
        self.bandit.update(ticket["state_key"], ticket["config"], reward)
        return reward
