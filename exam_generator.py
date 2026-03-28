"""
Exam generation logic — prompt construction, batched API calls, JSON parsing.
"""
from __future__ import annotations
import json
import re
import time
from typing import Callable

from llm_providers import call_llm

# ─── Topic definitions ────────────────────────────────────────────────────────

TOPICS: dict[str, list[str]] = {
    "Machine Learning": [
        "Supervised Learning — Regression (Linear, Polynomial, Ridge, Lasso)",
        "Supervised Learning — Classification (Logistic Regression, Naive Bayes, KNN)",
        "Support Vector Machines (SVM, kernels, margin maximization)",
        "Decision Trees, Random Forests & Ensemble Methods",
        "Unsupervised Learning — Clustering (K-Means, DBSCAN, Hierarchical)",
        "Unsupervised Learning — Dimensionality Reduction (PCA, t-SNE, UMAP)",
        "Boosting Algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM)",
        "Model Evaluation, Cross-Validation & Hyperparameter Tuning",
        "Feature Engineering & Selection",
        "Bias-Variance Tradeoff & Regularization (L1/L2)",
    ],
    "Deep Learning": [
        "Artificial Neural Networks & Backpropagation",
        "Deep Neural Networks (DNN) — Architecture & Activation Functions",
        "Convolutional Neural Networks (CNN) — Filters, Pooling, Architectures",
        "Recurrent Neural Networks (RNN) — Vanishing Gradient, BPTT",
        "Long Short-Term Memory (LSTM) & Gated Recurrent Units (GRU)",
        "Autoencoders & Variational Autoencoders (VAE)",
        "Generative Adversarial Networks (GAN)",
        "Optimization — SGD, Adam, Learning Rate Scheduling",
        "Regularization — Dropout, Batch Normalization, Weight Decay",
        "Transfer Learning & Fine-tuning",
    ],
    "LLMs & Generative AI": [
        "Transformer Architecture — Self-Attention, Multi-Head Attention",
        "Positional Encoding & Tokenization",
        "BERT, GPT, T5 and Foundational Model Families",
        "Pre-training Objectives (MLM, CLM, Seq2Seq)",
        "Fine-tuning, RLHF & Constitutional AI",
        "Prompt Engineering — Zero-shot, Few-shot, Chain-of-Thought",
        "Retrieval-Augmented Generation (RAG) & Vector Databases",
        "Hallucinations, Alignment & Model Evaluation",
        "AI Security — Jailbreaking, Prompt Injection, Adversarial Attacks",
        "Multimodal Models & AI Agents",
    ],
}

BATCH_SIZE = 20  # Questions per API call (stays well within context limits)

# ─── Public API ───────────────────────────────────────────────────────────────

Question = dict  # Type alias for a question dict


def generate_exam(
    provider: str,
    model: str,
    api_key: str,
    topics: list[str],
    num_questions: int,
    difficulties: list[str],
    progress_cb: Callable[[int, int, str], None] | None = None,
    groq_delay: float = 2.0,
) -> list[Question]:
    """
    Generate `num_questions` MCQ questions covering `topics` at the specified
    `difficulties`.  Large exams are split into batches of BATCH_SIZE.

    Args:
        provider: LLM provider name.
        model: Model identifier.
        api_key: Provider API key.
        topics: List of topic/subtopic strings.
        num_questions: Total questions to generate (1–100).
        difficulties: Non-empty list of "Easy", "Medium", "Hard".
        progress_cb: Optional callback(batch_done, total_batches, status_msg).
        groq_delay: Seconds to sleep between Groq batches (free-tier rate limit).

    Returns:
        List of question dicts.
    """
    topics_str = "\n".join(f"  - {t}" for t in topics)
    diff_instruction = _build_difficulty_instruction(difficulties)
    json_mode = provider in ("OpenAI", "Groq", "Groq (Free)")

    all_questions: list[Question] = []
    total_batches = (num_questions + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        remaining = num_questions - len(all_questions)
        batch_n = min(BATCH_SIZE, remaining)
        id_offset = len(all_questions)

        if progress_cb:
            progress_cb(
                batch_idx,
                total_batches,
                f"Generating batch {batch_idx + 1}/{total_batches} "
                f"(questions {id_offset + 1}–{id_offset + batch_n})…",
            )

        messages = _build_messages(topics_str, batch_n, diff_instruction, id_offset + 1)
        raw = call_llm(provider, model, api_key, messages, json_mode=json_mode)
        batch = _parse_response(raw)

        # Re-number to ensure globally unique, sequential IDs
        for i, q in enumerate(batch):
            q["id"] = id_offset + i + 1

        all_questions.extend(batch)

        # Respect Groq free-tier rate limits between batches
        if provider in ("Groq", "Groq (Free)") and batch_idx < total_batches - 1:
            time.sleep(groq_delay)

    if progress_cb:
        progress_cb(total_batches, total_batches, "Done!")

    return all_questions[:num_questions]


def generate_concept_response(
    provider: str,
    model: str,
    api_key: str,
    topic: str,
    history: list[dict],
    user_question: str,
) -> str:
    """
    Return a tutor-style explanation for `user_question` in the context of `topic`.

    Args:
        history: Previous messages in [{"role": ..., "content": ...}] format
                 (excluding the latest user question).
        user_question: The student's latest question.
    """
    system = (
        f"You are an expert AI/ML tutor specializing in {topic}. "
        "Provide clear, accurate, and educational explanations. "
        "Use concrete examples, analogies, and code snippets where helpful. "
        "Format your response with Markdown headings, bullet points, and code blocks."
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_question})
    return call_llm(provider, model, api_key, messages)


# ─── Prompt construction ──────────────────────────────────────────────────────

def _build_difficulty_instruction(difficulties: list[str]) -> str:
    if len(difficulties) == 1:
        d = difficulties[0]
        desc = {
            "Easy": "basic definitions, terminology, and recall of fundamental facts",
            "Medium": "conceptual understanding, application, and comparisons",
            "Hard": "deep analysis, edge cases, implementation nuances, and multi-step reasoning",
        }.get(d, d)
        return f"ALL questions must be {d} difficulty: {desc}."

    parts = []
    for d in difficulties:
        desc = {
            "Easy": "recall/definitions",
            "Medium": "understanding/application",
            "Hard": "analysis/edge-cases",
        }.get(d, d)
        parts.append(f"{d} ({desc})")
    return (
        f"Distribute difficulty EVENLY across: {', '.join(parts)}. "
        "Mix them throughout the list — do not group by difficulty."
    )


def _build_messages(
    topics_str: str,
    batch_n: int,
    diff_instruction: str,
    start_id: int,
) -> list[dict]:
    system = (
        "You are an expert educator and exam writer specializing in computer science, "
        "machine learning, and artificial intelligence. "
        "You create rigorous, unambiguous, high-quality multiple-choice questions. "
        "You ALWAYS respond with valid JSON only — no markdown fences, no extra text."
    )

    example = json.dumps(
        {
            "id": start_id,
            "topic": "Supervised Learning",
            "difficulty": "Medium",
            "question": "Which of the following best describes the bias-variance tradeoff?",
            "options": {
                "A": "A model with high bias underfits and high variance overfits the training data.",
                "B": "Bias and variance always move in the same direction as model complexity increases.",
                "C": "Regularization increases both bias and variance simultaneously.",
                "D": "A high-variance model always generalizes better to unseen data.",
            },
            "correct_answer": "A",
            "explanation": (
                "High bias → the model is too simple and underfits (misses signal). "
                "High variance → the model is too complex and overfits (fits noise). "
                "The tradeoff is about finding the complexity sweet-spot."
            ),
        },
        indent=2,
    )

    user = f"""Generate exactly {batch_n} multiple-choice exam questions.

TOPICS (cover proportionally):
{topics_str}

DIFFICULTY: {diff_instruction}

REQUIREMENTS:
- Each question has exactly 4 options: A, B, C, D
- Exactly one option is correct; the others are plausible distractors
- Questions must be academically rigorous and unambiguous
- IDs run from {start_id} to {start_id + batch_n - 1}

OUTPUT FORMAT — return ONLY a valid JSON array:
[
{example},
  ... (remaining {batch_n - 1} questions)
]

No markdown, no explanation, no text outside the JSON array."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ─── Response parsing ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> list[Question]:
    """
    Extract a list of question dicts from the LLM's raw text response.
    Handles: bare JSON array, wrapped {"questions": [...]}, markdown fences.
    """
    text = _strip_fences(raw.strip())

    # Attempt 1: direct parse
    try:
        data = json.loads(text)
        return _extract_list(data)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find first [...] in text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            return _extract_list(data)
        except json.JSONDecodeError:
            pass

    # Attempt 3: find first {...} (wrapped object)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return _extract_list(data)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse questions from the model's response. "
        "Try again — the model may have returned malformed JSON."
    )


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that some models add despite instructions."""
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first line (``` or ```json) and last ``` line
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.strip()


def _extract_list(data) -> list[Question]:
    """Accept either a bare list or a dict with a 'questions' key."""
    if isinstance(data, list):
        return [_normalize(q) for q in data]
    if isinstance(data, dict):
        for key in ("questions", "exam", "items", "data"):
            if key in data and isinstance(data[key], list):
                return [_normalize(q) for q in data[key]]
    raise ValueError("JSON structure is not a question list.")


def _normalize(q: dict) -> dict:
    """Ensure required fields exist with sensible defaults."""
    return {
        "id": int(q.get("id", 0)),
        "topic": str(q.get("topic", "General")),
        "difficulty": str(q.get("difficulty", "Medium")).capitalize(),
        "question": str(q.get("question", "")),
        "options": {
            k: str(v)
            for k, v in q.get("options", {}).items()
            if k in ("A", "B", "C", "D")
        },
        "correct_answer": str(q.get("correct_answer", "A")).upper(),
        "explanation": str(q.get("explanation", "")),
    }
