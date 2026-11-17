"""Checks on the ML pieces. No model server needed: the agent uses the offline template backend."""
import os
import random

os.environ.setdefault("TICKETIQ_LLM_BACKEND", "template")

from ticketiq.agent import ANSWER, ESCALATE_TO_HUMAN, enforce_escalation_policy  # noqa: E402
from ticketiq.bandit import EpsilonGreedyContextualBandit, compute_reward  # noqa: E402
from ticketiq.classifier import CATEGORIES, NaiveBayesTextClassifier, TicketClassifierService, load_tickets, stratified_split  # noqa: E402
from ticketiq.config import SETTINGS  # noqa: E402
from ticketiq.retrieval import KnowledgeRetriever  # noqa: E402
from ticketiq.sentiment import AspectSentimentAnalyzer  # noqa: E402
from ticketiq.triage import TicketIQ  # noqa: E402


def test_naive_bayes_learns_and_scores_probabilities():
    nb = NaiveBayesTextClassifier().train(["refund my invoice charge", "app crashes on login error",
                                           "refund the double charge", "error when the page loads"],
                                          ["billing", "technical", "billing", "technical"])
    assert nb.predict("please refund this charge") == "billing"
    label, confidence = nb.predict_with_confidence("crash error again")
    assert label == "technical" and 0.5 < confidence <= 1.0


def test_split_is_stratified_and_disjoint():
    tickets = load_tickets(SETTINGS.tickets_file)
    train, test = stratified_split(tickets, SETTINGS.test_split, SETTINGS.random_seed)
    assert not {id(t) for t in train} & {id(t) for t in test}
    assert {t.category for t in test} == set(CATEGORIES)


def test_held_out_accuracy_matches_the_report():
    ev = TicketClassifierService(SETTINGS.tickets_file, SETTINGS.test_split, SETTINGS.random_seed).train().evaluation
    assert ev["test_size"] == 40 and round(ev["accuracy"], 3) == 0.950 and round(ev["macro_f1"], 3) == 0.949


def test_aspect_sentiment_finds_a_negative_billing_aspect():
    aspects = AspectSentimentAnalyzer().analyse("Charged twice", "The billing is wrong and I am very angry about it.")
    assert any(a.aspect == "billing" and a.score < 0 for a in aspects)


def test_retrieval_returns_the_refund_policy_for_a_refund_question():
    hits = KnowledgeRetriever(SETTINGS.knowledge_base_dir).load().retrieve("Refund", "I was charged twice", "billing", 2)
    assert len(hits) == 2 and "refund" in hits[0].as_dict()["source"].lower()


def test_bandit_converges_to_the_better_arm():
    rng = random.Random(0)
    bandit = EpsilonGreedyContextualBandit(["good", "bad"], epsilon=0.1, seed=0)
    for _ in range(500):
        action, _ = bandit.select_action("s")
        bandit.update("s", action, rng.gauss(1.0 if action == "good" else 0.0, 0.3))
    assert bandit.best_action("s") == "good"


def test_reward_prefers_helpful_and_fast():
    assert compute_reward(1, 1.0) > compute_reward(0, 1.0) and compute_reward(1, 1.0) > compute_reward(1, 5.0)


def test_feature_requests_are_never_escalated():
    assert enforce_escalation_policy(ESCALATE_TO_HUMAN, "feature_request")[0] == ANSWER


def test_end_to_end_ticket_and_feedback():
    app = TicketIQ()
    out = app.handle("Charged twice", "I was billed twice this month, please refund the duplicate.", "pro")
    assert out["category"] == "billing" and out["response_text"] and out["retrieved_knowledge"]
    assert isinstance(app.feedback(out, 1), float)
