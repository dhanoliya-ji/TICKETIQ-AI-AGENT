"""Contextual bandit: the pipeline configurations it chooses between, state, reward and the learner."""
import json
import random
import threading
from pathlib import Path

from .config import SETTINGS


# ==================================================================================================
# The pipeline configurations that the reinforcement learning layer chooses between.
# ==================================================================================================
# Each configuration ("arm" in bandit language) is a combination of:
#
# * a **prompt variant** - a distinct system instruction *and*, when a real
#   Ollama server is available, a distinct model.  The two variants really do
#   behave differently: one quotes policy tersely, the other explains step by
#   step and is noticeably slower because it writes more tokens.
# * a **RAG top-K** - how many knowledge base chunks are put in the context.
#   K=2 is fast and focused, K=5 is slower but covers edge cases.
#
# Two variants x two K values = four arms.  That is a small enough action space
# for a bandit to learn quickly from real feedback, and large enough that the
# choice genuinely matters: the best arm for a simple free-tier feature request
# is not the best arm for an angry enterprise outage.

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
CONCISE_POLICY = "concise_policy"
EMPATHETIC_STEPWISE = "empathetic_stepwise"

PROMPT_VARIANTS: dict[str, str] = {
    CONCISE_POLICY: (
        "You are TicketIQ, a support triage assistant for a B2B SaaS company. "
        "Answer in at most four sentences. Quote the exact policy rule you are "
        "relying on. Do not apologise, do not add pleasantries, do not invent "
        "any rule that is not in the provided context."
    ),
    EMPATHETIC_STEPWISE: (
        "You are TicketIQ, a support triage assistant for a B2B SaaS company. "
        "Open by acknowledging the impact on the customer in one sentence, then "
        "give clearly numbered next steps, then say what happens next and who "
        "owns it. Stay warm and concrete. Never invent a rule that is not in "
        "the provided context."
    ),
}


class PipelineConfig:
    """One complete choice of how to answer a ticket."""

    def __init__(self, prompt_variant: str, rag_top_k: int) -> None:
        self.prompt_variant = prompt_variant
        self.rag_top_k = rag_top_k

    @property
    def name(self) -> str:
        """Stable identifier used as the bandit action key and in the API response."""
        return self.prompt_variant + "|k" + str(self.rag_top_k)

    @property
    def system_prompt(self) -> str:
        return PROMPT_VARIANTS[self.prompt_variant]

    @property
    def model_name(self) -> str:
        """Which Ollama model this variant uses when a server is available."""
        if self.prompt_variant == CONCISE_POLICY:
            return SETTINGS.ollama_model_a
        return SETTINGS.ollama_model_b

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prompt_variant": self.prompt_variant,
            "rag_top_k": self.rag_top_k,
            "model": self.model_name,
        }


# The full action space, in a fixed order so that reports are comparable.
ALL_CONFIGS: list[PipelineConfig] = [
    PipelineConfig(CONCISE_POLICY, 2),
    PipelineConfig(CONCISE_POLICY, 5),
    PipelineConfig(EMPATHETIC_STEPWISE, 2),
    PipelineConfig(EMPATHETIC_STEPWISE, 5),
]

# Fast lookup from the action name back to the configuration object.
CONFIGS_BY_NAME: dict[str, PipelineConfig] = {}
for _config in ALL_CONFIGS:
    CONFIGS_BY_NAME[_config.name] = _config


def all_config_names() -> list[str]:
    """The bandit action space as a list of names."""
    names = []
    for config in ALL_CONFIGS:
        names.append(config.name)
    return names


def get_config(name: str) -> PipelineConfig:
    """Look up a configuration by name, with a clear error if it is unknown."""
    if name not in CONFIGS_BY_NAME:
        raise KeyError("unknown pipeline configuration: " + name)
    return CONFIGS_BY_NAME[name]


# ==================================================================================================
# The bandit state (the "context") and the reward function.
# ==================================================================================================
# State
# -----
# ``category | urgency bucket | customer tier`` = 4 x 3 x 3 = 36 possible states.
# All three are already discrete, and they are exactly the three things that
# should change which configuration we route to:
#
# * an enterprise customer deserves the careful, step-by-step answer;
# * a high urgency ticket justifies retrieving more context;
# * a feature request needs neither.
#
# Everything continuous (the raw urgency number, the sentiment scores) is
# bucketed before it reaches the bandit, because a bandit learns per state and
# continuous states would never repeat often enough to learn anything.
#
# Reward
# ------
# The assignment defines it exactly:
#
#     reward = (user feedback score x 10) - (latency in seconds)
#
# So a helpful answer is worth 10 points and every second of waiting costs one
# point.  An unhelpful answer scores a negative reward equal to its latency,
# which is what makes slow *and* unhelpful configurations get abandoned first.

# How much one unit of positive feedback is worth.
FEEDBACK_WEIGHT = 10.0


def build_state_key(category: str, urgency_bucket: str, customer_tier: str) -> str:
    """Combine the three context features into one dictionary key."""
    return category + "|" + urgency_bucket + "|" + customer_tier


def compute_reward(feedback_score: int, latency_seconds: float) -> float:
    """reward = feedback x 10 - latency."""
    return (feedback_score * FEEDBACK_WEIGHT) - latency_seconds


# ==================================================================================================
# An epsilon-greedy contextual bandit.
# ==================================================================================================
# Why a contextual bandit and not Q-learning or a policy network
# --------------------------------------------------------------
# Choosing a pipeline configuration is a *one step* decision: we pick an arm,
# we get a reward, and the next ticket is unrelated to this one.  There is no
# sequence of states to plan through, so the machinery of Q-learning (discount
# factors, next-state bootstrapping) would be dead weight.  A contextual bandit
# is the exact model for "one decision, immediate reward, and the right decision
# depends on the context".  It also learns from very few samples, which matters
# because every sample costs a real customer interaction.
#
# The rule
# --------
# For each (state, action) pair we keep a running average of the reward.  Then,
# for each new ticket:
#
# * with probability ``epsilon`` pick a random arm  -> **explore**;
# * otherwise pick the arm with the best average    -> **exploit**;
# * any arm never tried in this state is tried first, so no arm is written off
#   on zero evidence.
#
# The running average is updated incrementally:
#
#     new_average = old_average + (reward - old_average) / new_count
#
# which is the same number as re-averaging the whole history, without storing it.

# Value reported for a state/action pair that has never been tried.
NO_DATA = 0.0


class ArmStatistics:
    """How one action has performed in one state."""

    def __init__(self, pulls: int = 0, average_reward: float = 0.0) -> None:
        # How many times this arm was chosen *and* rewarded in this state.
        self.pulls = pulls
        self.average_reward = average_reward

    def update(self, reward: float) -> None:
        """Fold one new reward into the running average."""
        self.pulls = self.pulls + 1
        difference = reward - self.average_reward
        self.average_reward = self.average_reward + (difference / self.pulls)

    def as_dict(self) -> dict[str, float]:
        return {"pulls": self.pulls, "average_reward": round(self.average_reward, 4)}


class EpsilonGreedyContextualBandit:
    """Learns which action is best in each state, online, from rewards."""

    def __init__(self, actions: list[str], epsilon: float, seed: int = 0) -> None:
        if len(actions) == 0:
            raise ValueError("the bandit needs at least one action")
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0")

        self.actions = list(actions)
        self.epsilon = epsilon
        self.random_generator = random.Random(seed)

        # state key -> action name -> statistics
        self.statistics: dict[str, dict[str, ArmStatistics]] = {}

        # How many times each *reason* for a choice was used, which is what the
        # simulation report and the /rl/stats endpoint show.
        self.exploration_count = 0
        self.exploitation_count = 0

        # The API serves requests on several threads, so every mutation of the
        # statistics is guarded.
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Choosing an action
    # ------------------------------------------------------------------
    def select_action(self, state_key: str) -> tuple[str, str]:
        """Pick an action for this state.

        Returns ``(action name, why)`` where *why* is "explore", "exploit" or
        "cold_start", so the decision can be explained in the API response.
        """
        with self.lock:
            arms = self._arms_for_state(state_key)

            # 1. Cold start: try every arm once before trusting any average.
            untried_actions = []
            for action in self.actions:
                if arms[action].pulls == 0:
                    untried_actions.append(action)
            if len(untried_actions) > 0:
                chosen = self.random_generator.choice(untried_actions)
                self.exploration_count = self.exploration_count + 1
                return chosen, "cold_start"

            # 2. Explore with probability epsilon.
            if self.random_generator.random() < self.epsilon:
                chosen = self.random_generator.choice(self.actions)
                self.exploration_count = self.exploration_count + 1
                return chosen, "explore"

            # 3. Otherwise exploit the best known arm. Every arm has been
            #    pulled by now - the cold-start branch above guarantees it -
            #    but the shared helper keeps the two agreeing regardless.
            self.exploitation_count = self.exploitation_count + 1
            return self._best_known_action(arms), "exploit"

    # ------------------------------------------------------------------
    # Learning from a reward
    # ------------------------------------------------------------------
    def update(self, state_key: str, action: str, reward: float) -> None:
        """Record the reward an action earned in a state."""
        if action not in self.actions:
            raise KeyError("unknown action: " + action)

        with self.lock:
            arms = self._arms_for_state(state_key)
            arms[action].update(reward)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def best_action(self, state_key: str) -> str:
        """The arm with the highest average reward in this state, ignoring epsilon."""
        with self.lock:
            return self._best_known_action(self._arms_for_state(state_key))

    def _best_known_action(self, arms: dict[str, ArmStatistics]) -> str:
        """The best arm *among those actually tried*.

        The "actually tried" part matters. An arm that has never been pulled
        still carries the initial average of 0.0, and a real reward can easily
        be negative: the reward is ``feedback * 10 - latency``, so any answer
        slower than ten seconds scores below zero even when the customer said
        it helped. Against a local language model that is the normal case, not
        an edge case. Comparing on average alone therefore reported an untried
        arm as the best one - it "beat" the only arm with evidence purely by
        never having been measured.

        When nothing has been tried yet there is no best arm, and the first
        action is returned as an arbitrary but stable placeholder. Callers that
        care reach this only after the cold-start rule has pulled every arm.

        The caller already holds the lock.
        """
        best = self.actions[0]
        best_is_known = arms[best].pulls > 0

        for action in self.actions:
            arm = arms[action]
            if arm.pulls == 0:
                continue
            if not best_is_known or arm.average_reward > arms[best].average_reward:
                best = action
                best_is_known = True

        return best

    def total_pulls(self) -> int:
        with self.lock:
            total = 0
            for arms in self.statistics.values():
                for arm in arms.values():
                    total = total + arm.pulls
            return total

    def snapshot(self) -> dict[str, object]:
        """A JSON friendly view of everything the bandit has learned."""
        with self.lock:
            states: dict[str, dict[str, dict[str, float]]] = {}
            for state_key in sorted(self.statistics.keys()):
                arms = self.statistics[state_key]
                states[state_key] = {}
                for action in self.actions:
                    states[state_key][action] = arms[action].as_dict()

            return {
                "epsilon": self.epsilon,
                "actions": list(self.actions),
                "exploration_count": self.exploration_count,
                "exploitation_count": self.exploitation_count,
                "states": states,
            }

    # ------------------------------------------------------------------
    # Persistence, so learning survives a restart
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write the learned statistics to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Load statistics written by ``save``. Returns False if there is no file."""
        if not path.exists():
            return False

        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt state file should not stop the service from starting;
            # we simply begin learning again from scratch.
            return False

        with self.lock:
            self.exploration_count = int(stored.get("exploration_count", 0))
            self.exploitation_count = int(stored.get("exploitation_count", 0))

            self.statistics = {}
            stored_states = stored.get("states", {})
            for state_key, arms in stored_states.items():
                self.statistics[state_key] = {}
                for action in self.actions:
                    values = arms.get(action, {})
                    self.statistics[state_key][action] = ArmStatistics(
                        pulls=int(values.get("pulls", 0)),
                        average_reward=float(values.get("average_reward", NO_DATA)),
                    )
        return True

    # ------------------------------------------------------------------
    def _arms_for_state(self, state_key: str) -> dict[str, ArmStatistics]:
        """Return (creating if needed) the statistics for one state.

        The caller already holds the lock.
        """
        if state_key not in self.statistics:
            fresh: dict[str, ArmStatistics] = {}
            for action in self.actions:
                fresh[action] = ArmStatistics()
            self.statistics[state_key] = fresh
        return self.statistics[state_key]
