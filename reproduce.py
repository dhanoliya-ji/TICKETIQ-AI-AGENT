"""Reproduce every number in the README.

    python reproduce.py            # classifier report + 20,000-ticket bandit simulation
    python reproduce.py 5000       # shorter bandit run

1. Classifier: the from-scratch Naive Bayes trained on a stratified 75/25 split of data/tickets.json,
   scored with scikit-learn on the held-out tickets.
2. Bandit: a simulated world where each of the 4 pipeline configurations has a hidden helpfulness and
   latency per urgency bucket and tier. The epsilon-greedy contextual bandit only ever sees rewards; an
   oracle that knows the hidden numbers grades how often it picks the best configuration.
"""
import json
import random
import sys
from pathlib import Path

from ticketiq.bandit import EpsilonGreedyContextualBandit, all_config_names, build_state_key, compute_reward
from ticketiq.classifier import CATEGORIES, TicketClassifierService
from ticketiq.config import SETTINGS

RESULTS = Path(__file__).resolve().parent / "results"

# ------------------------------------------------------------------------------- 1. classifier
def classifier_report() -> dict:
    service = TicketClassifierService(SETTINGS.tickets_file, SETTINGS.test_split, SETTINGS.random_seed).train()
    ev = service.evaluation
    print(f"\nCLASSIFIER  {ev['training_size']} training / {ev['test_size']} held-out tickets")
    print(f"accuracy {ev['accuracy']:.3f}   macro precision {ev['macro_precision']:.3f}   "
          f"macro recall {ev['macro_recall']:.3f}   macro F1 {ev['macro_f1']:.3f}")
    print(f"{'category':18s}{'precision':>10s}{'recall':>8s}{'f1':>8s}{'support':>9s}")
    for c in CATEGORIES:
        m = ev["per_category"][c]
        print(f"{c:18s}{m['precision']:10.3f}{m['recall']:8.3f}{m['f1']:8.3f}{m['support']:9d}")
    print("confusion matrix (rows actual, columns predicted):")
    print(" " * 18 + "".join(f"{c[:9]:>10s}" for c in CATEGORIES))
    for a in CATEGORIES:
        print(f"{a:18s}" + "".join(f"{service.confusion[a][p]:10d}" for p in CATEGORIES))
    return {**{k: v for k, v in ev.items()}, "confusion": service.confusion}


# ------------------------------------------------------------------------------- 2. bandit simulation
SEED, EPSILON = 11, 0.15
URGENCY_BUCKETS, TIERS = ["low", "medium", "high"], ["free", "pro", "enterprise"]
SIM_CATEGORIES = ["billing", "technical", "account", "feature_request"]
# the hidden truth of the simulated world
BASE_LATENCY = {"concise_policy|k2": 1.0, "concise_policy|k5": 1.6, "empathetic_stepwise|k2": 2.2,
                "empathetic_stepwise|k5": 2.9}
HELPFULNESS = {  # probability the customer marks the answer helpful, by urgency bucket
    "low": {"concise_policy|k2": 0.90, "concise_policy|k5": 0.80, "empathetic_stepwise|k2": 0.55, "empathetic_stepwise|k5": 0.45},
    "medium": {"concise_policy|k2": 0.60, "concise_policy|k5": 0.85, "empathetic_stepwise|k2": 0.70, "empathetic_stepwise|k5": 0.80},
    "high": {"concise_policy|k2": 0.25, "concise_policy|k5": 0.45, "empathetic_stepwise|k2": 0.70, "empathetic_stepwise|k5": 0.95},
}
TIER_ADJUSTMENT = {"free": 0.04, "pro": 0.0, "enterprise": -0.04}


def simulate_latency(config: str, rng: random.Random) -> float:
    return max(0.2, BASE_LATENCY[config] + rng.gauss(0.0, 0.15))


def simulate_feedback(config: str, urgency: str, tier: str, rng: random.Random) -> int:
    return 1 if rng.random() < HELPFULNESS[urgency][config] + TIER_ADJUSTMENT[tier] else 0


def expected_reward(config: str, urgency: str, tier: str) -> float:
    return (HELPFULNESS[urgency][config] + TIER_ADJUSTMENT[tier]) * 10.0 - BASE_LATENCY[config]


def oracle(urgency: str, tier: str) -> str:
    best = all_config_names()[0]
    for a in all_config_names():
        if expected_reward(a, urgency, tier) > expected_reward(best, urgency, tier):
            best = a
    return best


def bandit_simulation(n: int = 20000, early: int = 1000) -> dict:
    rng = random.Random(SEED)
    bandit = EpsilonGreedyContextualBandit(all_config_names(), epsilon=EPSILON, seed=SEED)
    window, block = max(1, n // 5), max(1, n // 10)
    hits_early = hits_first = hits_last = 0
    total, block_total, blocks = 0.0, 0.0, []
    for t in range(n):
        category, urgency, tier = rng.choice(SIM_CATEGORIES), rng.choice(URGENCY_BUCKETS), rng.choice(TIERS)
        state = build_state_key(category, urgency, tier)
        action, _ = bandit.select_action(state)
        latency = simulate_latency(action, rng)
        feedback = simulate_feedback(action, urgency, tier, rng)
        reward = compute_reward(feedback, latency)
        bandit.update(state, action, reward)
        total += reward
        block_total += reward
        if (t + 1) % block == 0:
            blocks.append(block_total / block)
            block_total = 0.0
        best = action == oracle(urgency, tier)
        hits_early += best and t < early
        hits_first += best and t < window
        hits_last += best and t >= n - window
    res = {"tickets": n, "states": len(bandit.statistics), "epsilon": EPSILON,
           f"optimal_share_first_{early}": hits_early / min(early, n),
           f"optimal_share_first_{window}": hits_first / window,
           f"optimal_share_last_{window}": hits_last / window,
           "average_reward": total / n, "average_reward_by_block": blocks,
           "ceiling_with_exploration": 1 - EPSILON + EPSILON / len(all_config_names())}
    print(f"\nBANDIT  {n:,} simulated tickets, {res['states']} states, 4 configurations, epsilon {EPSILON}")
    print(f"picked the oracle's configuration: {100 * res[f'optimal_share_first_{early}']:.1f}% of the first {early:,} tickets, "
          f"{100 * res[f'optimal_share_first_{window}']:.1f}% of the first {window:,}, "
          f"{100 * res[f'optimal_share_last_{window}']:.1f}% of the last {window:,}")
    print(f"ceiling with 15% exploration: {100 * res['ceiling_with_exploration']:.1f}%   "
          f"average reward {blocks[0]:.2f} in the first block -> {blocks[-1]:.2f} in the last")
    return res


if __name__ == "__main__":
    out = {"classifier": classifier_report(), "bandit": bandit_simulation(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {RESULTS / 'results.json'}")
