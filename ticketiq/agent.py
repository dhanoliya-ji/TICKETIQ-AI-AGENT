"""ReAct agent: local LLM client (Ollama, with an offline template fallback), tools and the reasoning loop."""
import json
import logging
import requests
import time
from typing import Any

from .config import SETTINGS
from .bandit import PipelineConfig


# ==================================================================================================
# Talking to a language model, with a deterministic offline fallback.
# ==================================================================================================
# Two backends implement the same tiny interface:
#
# * ``OllamaBackend``   - posts to a local Ollama server (``llama3`` / ``mistral``).
# * ``TemplateBackend`` - writes the answer from the retrieved knowledge itself,
#   with no server at all.
#
# Why a fallback exists
# ---------------------
# Everything else in this project (the classifier, the retriever, the bandit,
# the workflow engine) must be testable and reviewable on a laptop or in CI
# where no model server is running.  ``TICKETIQ_LLM_BACKEND=auto`` therefore
# probes Ollama once at startup and quietly uses the template writer when there
# is no server.  The response always reports which backend produced it, so a
# reviewer can never mistake template output for model output.
#
# The template backend is *not* a stub that returns a fixed string: it reads the
# retrieved policy chunks, the chosen prompt variant and the tool results, and
# composes a different answer for each, so the rest of the pipeline - including
# the reward signal - stays meaningful offline.

logger = logging.getLogger("ticketiq.llm")

# The two things we ever ask a language model to do.
TASK_DECIDE = "decide"  # choose the next agent step, answer as JSON
TASK_WRITE = "write"  # write the customer facing reply

# A customer reply shorter than this is not a reply. The small Ollama models
# occasionally stop right after the salutation - one run in roughly ten came
# back as exactly "Dear [Customer]," - and it is not reproducible from the
# prompt, so it cannot be prompted away. Such a generation has failed, and the
# client already knows how to handle a failed generation: use the template
# backend for that one request. The decision task is exempt, because its whole
# answer is a small JSON object and is legitimately short.
MINIMUM_REPLY_CHARACTERS = 80


def is_usable_generation(text: str, task: str) -> bool:
    """Is this completion something we can actually send on?

    Empty output is never usable. Beyond that, only a customer reply has a
    length we can judge: see MINIMUM_REPLY_CHARACTERS above.
    """
    if text == "":
        return False
    if task == TASK_WRITE:
        return len(text) >= MINIMUM_REPLY_CHARACTERS
    return True


BACKEND_OLLAMA = "ollama"
BACKEND_TEMPLATE = "template"


class LlmRequest:
    """One call to a language model."""

    def __init__(self, system_prompt: str, user_prompt: str, task: str, facts: dict) -> None:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        # TASK_DECIDE or TASK_WRITE.
        self.task = task
        # Structured version of everything that is also written into the
        # prompt.  A real model ignores this and reads the prompt; the template
        # backend uses it so it does not have to parse its own prompt back.
        self.facts = facts


class LlmResponse:
    """What came back, plus how it was produced."""

    def __init__(self, text: str, backend: str, model: str, latency_seconds: float) -> None:
        self.text = text
        self.backend = backend
        self.model = model
        self.latency_seconds = latency_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 4),
        }


# ---------------------------------------------------------------------------
# Backend 1: a real Ollama server
# ---------------------------------------------------------------------------
class OllamaBackend:
    """Calls the ``/api/generate`` endpoint of a local Ollama server."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """One cheap probe: does the server answer its model listing endpoint?"""
        try:
            response = requests.get(self.base_url + "/api/tags", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, request: LlmRequest, model: str) -> str:
        """Send the prompt and return the raw completion text."""
        payload: dict[str, Any] = {
            "model": model,
            "system": request.system_prompt,
            "prompt": request.user_prompt,
            # We want one complete answer, not a token stream.
            "stream": False,
            "options": {"temperature": 0.2},
        }
        response = requests.post(
            self.base_url + "/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("response", "")).strip()


# ---------------------------------------------------------------------------
# Backend 2: the offline template writer
# ---------------------------------------------------------------------------
class TemplateBackend:
    """Composes an answer from the retrieved knowledge, with no model server."""

    def generate(self, request: LlmRequest, model: str) -> str:
        if request.task == TASK_DECIDE:
            return self._decide(request.facts)
        return self._write(request.facts)

    # -- deciding the next agent step ---------------------------------------
    def _decide(self, facts: dict) -> str:
        """Return the same JSON shape a real model is asked to produce.

        The rules mirror the escalation document: feature requests are never
        escalated, refunds are checked before they are promised, access
        problems need the account status, and an angry high urgency enterprise
        technical ticket goes to a human.
        """
        category = facts.get("category", "technical")
        urgency_bucket = facts.get("urgency_bucket", "low")
        customer_tier = facts.get("customer_tier", "free")
        text = str(facts.get("subject", "") + " " + facts.get("body", "")).lower()
        already_used_tools = facts.get("tools_already_used", [])

        def decision(thought: str, action: str, action_input: dict) -> str:
            return json.dumps({"thought": thought, "action": action, "action_input": action_input})

        # 1. A refund claim must be checked against the policy first.
        wants_refund = "refund" in text or "charged twice" in text or "duplicate" in text
        if (
            category == "billing"
            and wants_refund
            and "check_refund_eligibility" not in already_used_tools
        ):
            return decision(
                "The customer is asking for money back, so eligibility must be "
                "confirmed before anything is promised.",
                "check_refund_eligibility",
                {"order_id": facts.get("order_id", "unknown")},
            )

        # 2. An access problem needs the real account state.
        if category == "account" and "check_account_status" not in already_used_tools:
            return decision(
                "This is an access problem, so the current state of the account "
                "decides which instructions are correct.",
                "check_account_status",
                {"customer_id": facts.get("customer_id", "unknown")},
            )

        # 3. Serious technical pain on a paying tier goes to a specialist.
        if category == "technical" and urgency_bucket == "high" and customer_tier == "enterprise":
            return decision(
                "An enterprise customer reports a high urgency technical failure, "
                "which the escalation rules send to a human specialist.",
                "escalate_to_human",
                {"reason": "enterprise high urgency technical issue"},
            )

        # 4. Everything else can be answered from the retrieved policy.
        return decision(
            "The retrieved knowledge base sections cover this question, so it "
            "can be answered directly.",
            "answer",
            {},
        )

    # -- writing the customer reply -----------------------------------------
    def _write(self, facts: dict) -> str:
        """Compose the reply out of the retrieved chunks and the tool results."""
        prompt_variant = facts.get("prompt_variant", "concise_policy")
        snippets = facts.get("snippets", [])
        tool_results = facts.get("tool_results", [])
        decision = facts.get("decision", "answer")
        category = facts.get("category", "technical")

        # The policy lines we are allowed to rely on.
        policy_lines: list[str] = []
        for snippet in snippets:
            heading = snippet.get("heading", "")
            text = snippet.get("text", "")
            policy_lines.append(heading + ": " + _first_sentence(text))

        observation_lines: list[str] = []
        for result in tool_results:
            observation_lines.append(
                result.get("tool", "tool") + " -> " + result.get("summary", "")
            )

        if prompt_variant == "concise_policy":
            return self._write_concise(category, decision, policy_lines, observation_lines)
        return self._write_stepwise(category, decision, policy_lines, observation_lines)

    def _write_concise(
        self,
        category: str,
        decision: str,
        policy_lines: list[str],
        observation_lines: list[str],
    ) -> str:
        parts = ["Category: " + category + "."]

        if len(policy_lines) > 0:
            parts.append("Applicable policy - " + " ".join(policy_lines[:2]))
        else:
            parts.append("No matching policy section was found for this request.")

        for line in observation_lines:
            parts.append("Checked: " + line + ".")

        if decision == "escalate_to_human":
            parts.append("This ticket is escalated to a human specialist.")
        else:
            parts.append("Applying the rule above resolves this request.")

        return " ".join(parts)

    def _write_stepwise(
        self,
        category: str,
        decision: str,
        policy_lines: list[str],
        observation_lines: list[str],
    ) -> str:
        lines = ["I understand this is disruptive, and I am sorry for the trouble it is causing."]
        lines.append("")
        lines.append("Here is what applies to your " + category + " request:")

        step_number = 1
        for policy_line in policy_lines[:3]:
            lines.append(str(step_number) + ". " + policy_line)
            step_number = step_number + 1

        if len(policy_lines) == 0:
            lines.append("1. I could not find a policy section that covers this exactly.")

        for observation in observation_lines:
            lines.append(
                str(step_number) + ". I already checked this for you: " + observation + "."
            )
            step_number = step_number + 1

        lines.append("")
        if decision == "escalate_to_human":
            lines.append(
                "What happens next: a support specialist now owns this ticket and "
                "will come back to you with a fix or a status update."
            )
        else:
            lines.append(
                "What happens next: follow the steps above and reply to this "
                "ticket if anything is still not working."
            )

        return "\n".join(lines)


def _first_sentence(text: str) -> str:
    """Return the first sentence of a chunk, so replies stay short."""
    stripped = text.strip().replace("\n", " ")
    position = stripped.find(". ")
    if position == -1:
        return stripped
    return stripped[: position + 1]


# ---------------------------------------------------------------------------
# The client the rest of the application uses
# ---------------------------------------------------------------------------
class LlmClient:
    """Chooses a backend once and then serves every generation request."""

    def __init__(self) -> None:
        self.ollama = OllamaBackend(SETTINGS.ollama_base_url, SETTINGS.llm_timeout_seconds)
        self.template = TemplateBackend()
        self.active_backend = self._choose_backend()

    def _choose_backend(self) -> str:
        """Decide once, at startup, which backend to use."""
        configured = SETTINGS.llm_backend

        if configured == BACKEND_TEMPLATE:
            return BACKEND_TEMPLATE
        if configured == BACKEND_OLLAMA:
            return BACKEND_OLLAMA
        # "auto": use Ollama only if it actually answers.
        if self.ollama.is_available():
            return BACKEND_OLLAMA
        return BACKEND_TEMPLATE

    def generate(self, request: LlmRequest, model: str) -> LlmResponse:
        """Run one generation and always return a usable response.

        If the Ollama call fails at request time (server stopped, model not
        pulled, timeout) we fall back to the template backend for that single
        request rather than failing the whole ticket.
        """
        started_at = time.perf_counter()

        if self.active_backend == BACKEND_OLLAMA:
            try:
                text = self.ollama.generate(request, model)
                if is_usable_generation(text, request.task):
                    elapsed = time.perf_counter() - started_at
                    return LlmResponse(text, BACKEND_OLLAMA, model, elapsed)
                logger.warning(
                    "discarding a %s generation of %s characters from %s; "
                    "writing the reply with the template backend instead",
                    request.task,
                    len(text),
                    model,
                )
            except (requests.RequestException, ValueError):
                # Fall through to the template backend below.
                pass

        text = self.template.generate(request, model)
        elapsed = time.perf_counter() - started_at
        return LlmResponse(text, BACKEND_TEMPLATE, model, elapsed)


# ==================================================================================================
# Mock back-office tools the agent may call.
# ==================================================================================================
# In a real deployment these would call the billing system and the identity
# service.  Here they are deterministic fakes: the answer is derived from the
# identifier itself, so the same order id always gives the same result and tests
# never need a network.  Every tool returns a dictionary with a ``summary`` key,
# which is the single line fed back into the reasoning loop as an observation.

# Tool names, used in prompts, logs and the API response.
CHECK_ACCOUNT_STATUS = "check_account_status"
CHECK_REFUND_ELIGIBILITY = "check_refund_eligibility"

# Non-tool actions the agent may finish with.
ANSWER = "answer"
ESCALATE_TO_HUMAN = "escalate_to_human"

ALL_TOOL_NAMES = [CHECK_ACCOUNT_STATUS, CHECK_REFUND_ELIGIBILITY]
ALL_ACTION_NAMES = ALL_TOOL_NAMES + [ANSWER, ESCALATE_TO_HUMAN]


class ToolResult:
    """The outcome of one tool call."""

    def __init__(self, tool: str, arguments: dict, output: dict, summary: str) -> None:
        self.tool = tool
        self.arguments = arguments
        self.output = output
        self.summary = summary

    def as_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "output": self.output,
            "summary": self.summary,
        }


def _stable_number(identifier: str) -> int:
    """Turn any identifier into a stable number between 0 and 99.

    ``hash()`` is not used because Python randomises it between processes,
    which would make the fake answers change from run to run.  Summing the
    character codes is crude but perfectly stable.
    """
    total = 0
    for character in str(identifier):
        total = total + ord(character)
    return total % 100


def check_account_status(arguments: dict) -> ToolResult:
    """Look up whether an account is active, locked or suspended."""
    customer_id = str(arguments.get("customer_id", "unknown"))
    number = _stable_number(customer_id)

    if number < 60:
        status = "active"
    elif number < 85:
        status = "locked"
    else:
        status = "suspended"

    if status == "active":
        plan = "pro"
    else:
        plan = "free"

    output = {
        "customer_id": customer_id,
        "status": status,
        "plan": plan,
        "failed_login_attempts": number % 12,
        "two_factor_enabled": number % 2 == 0,
    }
    summary = (
        "account "
        + customer_id
        + " is "
        + status
        + " on the "
        + plan
        + " plan with "
        + str(output["failed_login_attempts"])
        + " recent failed sign in attempts"
    )
    return ToolResult(CHECK_ACCOUNT_STATUS, arguments, output, summary)


def check_refund_eligibility(arguments: dict) -> ToolResult:
    """Check whether an order can still be refunded under the 30 day policy."""
    order_id = str(arguments.get("order_id", "unknown"))
    number = _stable_number(order_id)

    days_since_charge = number % 45
    amount = round(19.0 + (number * 7.5), 2)
    is_duplicate = number % 4 == 0

    # Mirrors the refund policy document: inside 30 days, or a duplicate
    # charge at any age.
    eligible = days_since_charge <= 30 or is_duplicate

    if eligible:
        if is_duplicate:
            reason = "duplicate charge, refundable regardless of age"
        else:
            reason = "within the 30 day refund window"
    else:
        reason = "charge is " + str(days_since_charge) + " days old, outside the 30 day window"

    output = {
        "order_id": order_id,
        "eligible": eligible,
        "amount": amount,
        "days_since_charge": days_since_charge,
        "duplicate_charge": is_duplicate,
        "reason": reason,
    }
    summary = (
        "order "
        + order_id
        + " is "
        + ("eligible" if eligible else "not eligible")
        + " for a refund of "
        + str(amount)
        + " ("
        + reason
        + ")"
    )
    return ToolResult(CHECK_REFUND_ELIGIBILITY, arguments, output, summary)


# Name -> function, so the agent can look a tool up by the string the model
# produced.
TOOL_REGISTRY = {
    CHECK_ACCOUNT_STATUS: check_account_status,
    CHECK_REFUND_ELIGIBILITY: check_refund_eligibility,
}


def run_tool(name: str, arguments: dict) -> ToolResult:
    """Execute a tool by name, returning a clear error result if it is unknown."""
    if name not in TOOL_REGISTRY:
        return ToolResult(
            tool=name,
            arguments=arguments,
            output={"error": "unknown tool"},
            summary="tool " + name + " does not exist",
        )
    return TOOL_REGISTRY[name](arguments)


# ==================================================================================================
# A ReAct style reasoning loop.
# ==================================================================================================
# ReAct = *Reason* then *Act*.  On every turn the model is shown the ticket, the
# retrieved policy and everything observed so far, and must answer with one JSON
# object:
#
#     {"thought": "...", "action": "...", "action_input": {...}}
#
# ``action`` is one of:
#
# * ``check_account_status``     - call the identity tool, then think again.
# * ``check_refund_eligibility`` - call the billing tool, then think again.
# * ``answer``                   - enough is known, write the reply.
# * ``escalate_to_human``        - hand the ticket to a specialist.
#
# Tool actions feed their result back as an observation and the loop runs again.
# Terminal actions end the loop and a second model call writes the customer
# reply.  ``agent_max_steps`` is a hard stop so a confused model can never spin.
#
# Everything the loop did - each thought, action and observation - is kept in
# ``AgentResult.trace`` and returned by the API, because a triage decision that
# cannot be audited is not worth much in support operations.

# The assignment asks for the agent's decision, its tool calls and its
# reasoning trace to be inspectable in the logs as well as in the API response.
# Logging from the agent itself - rather than from the endpoint - means the
# trace is recorded even when the agent is driven by a script or a test.
logger = logging.getLogger("ticketiq.agent")

# The one category whose escalation the knowledge base forbids outright.
FEATURE_REQUEST_CATEGORY = "feature_request"

# How many times the model may re-ask for a call it has already made before the
# loop stops humouring it and answers with the result already in hand.
REPEATS_BEFORE_GIVING_UP = 2

FEATURE_REQUEST_OVERRIDE_NOTE = (
    "policy: a feature request is never escalated, so this was answered on the "
    "normal queue instead"
)


def enforce_escalation_policy(decision: str, category: str) -> tuple[str, str]:
    """Apply the one escalation rule the knowledge base states outright.

    Returns ``(decision, override note)``; the note is empty when nothing was
    changed. This is a function rather than an inline check because three
    separate paths can end in an escalation - the model choosing it, an
    unparseable reply, and the step limit running out - and the rule has to
    hold on all three, not just the tidy one.
    """
    if decision == ESCALATE_TO_HUMAN and category == FEATURE_REQUEST_CATEGORY:
        return ANSWER, FEATURE_REQUEST_OVERRIDE_NOTE
    return decision, ""


class AgentStep:
    """One turn of the loop, kept for the audit trail."""

    def __init__(self, number: int, thought: str, action: str, action_input: dict) -> None:
        self.number = number
        self.thought = thought
        self.action = action
        self.action_input = action_input
        # Filled in only when the action was a tool call.
        self.observation: str = ""
        # Set when a policy rule changed the model's chosen action. Empty for
        # the ordinary case, so the trace shows plainly when the system
        # overrode the model and why.
        self.override: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "override": self.override,
        }


class AgentResult:
    """Everything the agent produced for one ticket."""

    def __init__(self) -> None:
        self.decision: str = ANSWER
        self.response_text: str = ""
        self.trace: list[AgentStep] = []
        self.tool_results: list[ToolResult] = []
        self.llm_calls: int = 0
        self.llm_model: str = ""
        # Which backend served each generation, in order. A single ticket can
        # legitimately use both: if a model call times out the client falls
        # back to the template writer for that request only.
        self.backends_used: list[str] = []

    @property
    def llm_backend(self) -> str:
        """The backend that served this ticket.

        Reporting only the last call would have been misleading: a ticket whose
        decision fell back to the template but whose reply was written by the
        model used to report "template", which undersold what actually
        happened. When the calls disagree, both names are reported.
        """
        distinct: list[str] = []
        for backend in self.backends_used:
            if backend not in distinct:
                distinct.append(backend)

        if len(distinct) == 0:
            return ""
        return "+".join(distinct)

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "reasoning_trace": [step.as_dict() for step in self.trace],
            "tool_calls": [result.as_dict() for result in self.tool_results],
            "llm_calls": self.llm_calls,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def extract_identifiers(subject: str, body: str) -> dict[str, str]:
    """Pull an order id and a customer id out of the ticket text.

    Support tickets rarely carry structured fields, so we look for the first
    run of digits after the words "order" or "account"/"customer".  When
    nothing is found the tools still run and report on an unknown identifier,
    which is exactly what a real agent would surface to the human.
    """
    words = (subject + " " + body).replace(",", " ").replace(".", " ").split()

    order_id = "unknown"
    customer_id = "unknown"

    for index in range(len(words)):
        word = words[index].lower()
        digits = "".join(character for character in words[index] if character.isdigit())

        if digits == "":
            continue

        # A number directly after one of these words identifies the thing.
        previous_word = words[index - 1].lower() if index > 0 else ""
        if previous_word in ("order", "invoice", "payment", "charge"):
            order_id = digits
        elif previous_word in ("account", "customer", "workspace", "user"):
            customer_id = digits
        elif order_id == "unknown" and word.startswith("ord"):
            order_id = digits

    return {"order_id": order_id, "customer_id": customer_id}


def parse_first_json_object(text: str) -> dict | None:
    """Find and parse the first ``{...}`` object in a model reply.

    Models often wrap JSON in prose or a code fence, so we scan for balanced
    braces instead of trusting the whole string to be valid JSON.
    """
    start_index = text.find("{")
    if start_index == -1:
        return None

    depth = 0
    for index in range(start_index, len(text)):
        character = text[index]
        if character == "{":
            depth = depth + 1
        elif character == "}":
            depth = depth - 1
            if depth == 0:
                candidate = text[start_index : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
    return None


def format_snippets(snippets: list[dict]) -> str:
    """Render the retrieved chunks as numbered context for the prompt.

    Snippets are plain dictionaries rather than objects because the workflow
    engine persists every stage output as JSON, and the output of the
    retrieval stage is exactly what is handed to the agent here.
    """
    if len(snippets) == 0:
        return "(no relevant knowledge base section was found)"

    lines = []
    for position in range(len(snippets)):
        snippet = snippets[position]
        lines.append(
            "["
            + str(position + 1)
            + "] "
            + str(snippet.get("document_title", ""))
            + " / "
            + str(snippet.get("heading", ""))
            + "\n"
            + str(snippet.get("text", ""))
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class TriageAgent:
    """Runs the reason/act loop and then writes the customer reply."""

    def __init__(self, llm_client: LlmClient) -> None:
        self.llm = llm_client
        self.max_steps = SETTINGS.agent_max_steps

    def run(
        self,
        subject: str,
        body: str,
        customer_tier: str,
        category: str,
        urgency_bucket: str,
        snippets: list[dict],
        config: PipelineConfig,
    ) -> AgentResult:
        """Decide what to do about one ticket and produce the reply text."""
        result = AgentResult()
        result.llm_model = config.model_name

        identifiers = extract_identifiers(subject, body)
        knowledge_text = format_snippets(snippets)

        tools_already_used: list[str] = []
        observations: list[str] = []
        # tool + arguments -> the summary it returned, so an identical repeat
        # can be answered from memory instead of run again.
        completed_calls: dict[str, str] = {}
        # How many times each call has been asked for again after that.
        repeat_counts: dict[str, int] = {}
        decision = ANSWER

        for step_number in range(1, self.max_steps + 1):
            facts = {
                "category": category,
                "urgency_bucket": urgency_bucket,
                "customer_tier": customer_tier,
                "subject": subject,
                "body": body,
                "order_id": identifiers["order_id"],
                "customer_id": identifiers["customer_id"],
                "tools_already_used": list(tools_already_used),
            }
            request = LlmRequest(
                system_prompt=self._decision_system_prompt(),
                user_prompt=self._decision_user_prompt(
                    subject,
                    body,
                    customer_tier,
                    category,
                    urgency_bucket,
                    knowledge_text,
                    observations,
                    identifiers,
                ),
                task=TASK_DECIDE,
                facts=facts,
            )

            response = self.llm.generate(request, config.model_name)
            result.llm_calls = result.llm_calls + 1
            result.backends_used.append(response.backend)

            parsed = parse_first_json_object(response.text)
            if parsed is None:
                # An unparseable reply is treated as "I cannot decide", which
                # is safest resolved by a human rather than by guessing.
                step = AgentStep(
                    step_number,
                    "The model reply could not be parsed as a decision.",
                    ESCALATE_TO_HUMAN,
                    {"reason": "unparseable model response"},
                )
                result.trace.append(step)
                decision = ESCALATE_TO_HUMAN
                break

            thought = str(parsed.get("thought", "")).strip()
            action = str(parsed.get("action", ANSWER)).strip()
            action_input = parsed.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            # An action we do not recognise is downgraded to a plain answer.
            if action not in ALL_ACTION_NAMES:
                action = ANSWER

            # One policy rule is enforced rather than merely asked for. The
            # knowledge base states twice - in the feature request document and
            # in the escalation rules - that a feature request is never
            # escalated, and the system prompt says so too, but a small model
            # ignores it: llama3.2:1b escalated all four sample categories,
            # including a request for dark mode. Quoting a policy to customers
            # while acting against it is not a defensible default, so the
            # decision is corrected here and the correction is recorded in the
            # trace rather than hidden.
            action, override_note = enforce_escalation_policy(action, category)
            if override_note:
                logger.info("step %s | policy override | %s", step_number, override_note)

            step = AgentStep(step_number, thought, action, action_input)
            step.override = override_note
            result.trace.append(step)

            logger.info(
                "step %s | action=%s | input=%s | thought=%s",
                step_number,
                action,
                json.dumps(action_input),
                thought,
            )

            if action in ALL_TOOL_NAMES:
                # Fill in an identifier the model forgot to pass through.
                if action == "check_refund_eligibility" and "order_id" not in action_input:
                    action_input["order_id"] = identifiers["order_id"]
                if action == "check_account_status" and "customer_id" not in action_input:
                    action_input["customer_id"] = identifiers["customer_id"]

                # Smaller models sometimes ask for a call they have already
                # made. Re-running it would burn a step, and for a tool with
                # real side effects it would be worse than wasteful, so the
                # earlier answer is replayed instead.
                call_signature = action + " " + json.dumps(action_input, sort_keys=True)

                if call_signature in completed_calls:
                    previous_summary = completed_calls[call_signature]
                    step.observation = previous_summary + " (already checked at an earlier step)"
                    repeat_counts[call_signature] = repeat_counts.get(call_signature, 0) + 1

                    logger.info(
                        "step %s | tool=%s | repeated call, replaying the earlier result",
                        step_number,
                        action,
                    )

                    # Replaying once gives the model another chance to use the
                    # answer it already has. Asking a third time means it is not
                    # going to, and every further turn is a wasted model call:
                    # llama3.2:1b spent five turns and 19.5 seconds looping on
                    # check_account_status before the step limit escalated it,
                    # which also drags the reward down, since reward subtracts
                    # latency. The tool result and the retrieved policy are
                    # already in hand, so the loop ends with an answer instead.
                    if repeat_counts[call_signature] >= REPEATS_BEFORE_GIVING_UP:
                        step.action = ANSWER
                        step.override = (
                            "the model asked for the same tool call "
                            + str(repeat_counts[call_signature] + 1)
                            + " times; answering with the result it already had"
                        )
                        logger.info("step %s | %s", step_number, step.override)
                        decision = ANSWER
                        break

                    observations.append(action + " -> " + previous_summary)
                    continue

                tool_result = run_tool(action, action_input)
                result.tool_results.append(tool_result)
                step.observation = tool_result.summary
                completed_calls[call_signature] = tool_result.summary

                logger.info(
                    "step %s | tool=%s | arguments=%s | observation=%s",
                    step_number,
                    tool_result.tool,
                    json.dumps(tool_result.arguments),
                    tool_result.summary,
                )

                tools_already_used.append(action)
                observations.append(action + " -> " + tool_result.summary)
                # Loop again so the model can reason about what it just learned.
                continue

            # A terminal action ends the loop.
            decision = action
            break
        else:
            # The loop ran out of steps without a terminal action.
            decision = ESCALATE_TO_HUMAN
            result.trace.append(
                AgentStep(
                    self.max_steps + 1,
                    "The step limit was reached without a conclusion.",
                    ESCALATE_TO_HUMAN,
                    {"reason": "step limit reached"},
                )
            )

        # The two paths above - an unparseable reply and the step limit - set the
        # decision directly, so the rule is applied once more here. Without
        # this, a feature request whose model reply failed to parse would be
        # escalated in defiance of the policy the system quotes to customers.
        decision, final_override = enforce_escalation_policy(decision, category)
        if final_override:
            logger.info("policy override | %s", final_override)
            if len(result.trace) > 0:
                result.trace[-1].action = decision
                result.trace[-1].override = final_override

        logger.info(
            "decision=%s | steps=%s | tools=%s | config=%s",
            decision,
            len(result.trace),
            json.dumps([tool.tool for tool in result.tool_results]),
            config.name,
        )

        result.decision = decision

        written = self._write_reply(
            subject, body, customer_tier, category, decision, snippets, result.tool_results, config
        )
        result.response_text = written.text
        result.backends_used.append(written.backend)
        result.llm_calls = result.llm_calls + 1
        return result

    # -- prompt building ----------------------------------------------------
    def _decision_system_prompt(self) -> str:
        return (
            "You are the reasoning core of a support triage agent. "
            "Reply with exactly one JSON object and no other text. "
            'The object has the keys "thought", "action" and "action_input". '
            '"action" must be one of: ' + ", ".join(ALL_ACTION_NAMES) + ". "
            "Call a tool only when its result would change your reply. "
            "Never escalate a feature request."
        )

    def _decision_user_prompt(
        self,
        subject: str,
        body: str,
        customer_tier: str,
        category: str,
        urgency_bucket: str,
        knowledge_text: str,
        observations: list[str],
        identifiers: dict[str, str],
    ) -> str:
        if len(observations) == 0:
            observation_text = "(no tools have been called yet)"
        else:
            observation_text = "\n".join(observations)

        return (
            "TICKET SUBJECT: " + subject + "\n"
            "TICKET BODY: " + body + "\n"
            "PREDICTED CATEGORY: " + category + "\n"
            "URGENCY: " + urgency_bucket + "\n"
            "CUSTOMER TIER: " + customer_tier + "\n"
            "ORDER ID: " + identifiers["order_id"] + "\n"
            "CUSTOMER ID: " + identifiers["customer_id"] + "\n\n"
            "RETRIEVED KNOWLEDGE BASE SECTIONS:\n" + knowledge_text + "\n\n"
            "OBSERVATIONS SO FAR:\n" + observation_text + "\n\n"
            "What is the next action?"
        )

    def _write_reply(
        self,
        subject: str,
        body: str,
        customer_tier: str,
        category: str,
        decision: str,
        snippets: list[dict],
        tool_results: list[ToolResult],
        config: PipelineConfig,
    ) -> LlmResponse:
        """Second model call: turn the decision into the customer facing text.

        Returns the whole response, not just the text, so the caller can record
        which backend actually wrote the reply.
        """
        tool_summaries = []
        for tool_result in tool_results:
            tool_summaries.append({"tool": tool_result.tool, "summary": tool_result.summary})

        facts = {
            "prompt_variant": config.prompt_variant,
            "category": category,
            "customer_tier": customer_tier,
            "decision": decision,
            "snippets": snippets,
            "tool_results": tool_summaries,
        }

        if decision == ESCALATE_TO_HUMAN:
            instruction = (
                "This ticket is being escalated to a human specialist. Tell the "
                "customer what will happen next without promising a fix."
            )
        else:
            instruction = "Answer the customer using only the knowledge below."

        user_prompt = (
            instruction + "\n\n"
            "TICKET SUBJECT: " + subject + "\n"
            "TICKET BODY: " + body + "\n"
            "CUSTOMER TIER: " + customer_tier + "\n"
            "CATEGORY: " + category + "\n\n"
            "KNOWLEDGE BASE:\n" + format_snippets(snippets) + "\n\n"
            "TOOL RESULTS:\n" + self._format_tool_results(tool_results) + "\n\n"
            "Write the reply to the customer now."
        )

        request = LlmRequest(
            system_prompt=config.system_prompt,
            user_prompt=user_prompt,
            task=TASK_WRITE,
            facts=facts,
        )
        response = self.llm.generate(request, config.model_name)
        return response

    def _format_tool_results(self, tool_results: list[ToolResult]) -> str:
        if len(tool_results) == 0:
            return "(no tools were called)"
        lines = []
        for tool_result in tool_results:
            lines.append(tool_result.tool + ": " + tool_result.summary)
        return "\n".join(lines)
