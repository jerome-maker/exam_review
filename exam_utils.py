"""
Exam formatting and export utilities — no Streamlit dependencies.
"""
from __future__ import annotations
from datetime import datetime


Question = dict  # type alias


# ─── Text export ──────────────────────────────────────────────────────────────

def format_student_exam(questions: list[Question], title: str) -> str:
    """Plain-text exam for students — no answers or explanations."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 62,
        title.upper(),
        f"Generated : {ts}",
        f"Questions : {len(questions)}",
        "=" * 62,
        "",
        "INSTRUCTIONS",
        "  • Read each question carefully.",
        "  • Select the single best answer (A, B, C, or D).",
        "  • Each question is worth equal marks.",
        "",
        "-" * 62,
        "",
    ]

    for q in questions:
        diff_tag = f"[{q.get('difficulty', '')}]"
        topic_tag = f"[{q.get('topic', '')}]"
        lines.append(f"Question {q['id']}  {diff_tag}  {topic_tag}")
        lines.append(q["question"])
        lines.append("")
        for letter in ("A", "B", "C", "D"):
            opt = q.get("options", {}).get(letter, "")
            if opt:
                lines.append(f"    {letter})  {opt}")
        lines.append("")
        lines.append("    Answer: ____")
        lines.append("")

    lines += ["-" * 62, f"  End of exam — {len(questions)} questions total", "-" * 62]
    return "\n".join(lines)


def format_answer_key(questions: list[Question], title: str) -> str:
    """Plain-text answer key with explanations for instructors."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 62,
        f"{title.upper()}  —  ANSWER KEY  (INSTRUCTOR COPY)",
        f"Generated : {ts}",
        f"Questions : {len(questions)}",
        "=" * 62,
        "",
    ]

    for q in questions:
        correct = q.get("correct_answer", "?")
        answer_text = q.get("options", {}).get(correct, "")
        explanation = q.get("explanation", "")

        lines.append(
            f"Q{q['id']:>3}.  [{q.get('topic', '')}]  [{q.get('difficulty', '')}]"
        )
        lines.append(f"       Answer: {correct})  {answer_text}")
        if explanation:
            lines.append(f"  Explanation: {explanation}")
        lines.append("")

    # Summary table
    diff_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    for q in questions:
        d = q.get("difficulty", "Unknown")
        t = q.get("topic", "Unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        topic_counts[t] = topic_counts.get(t, 0) + 1

    lines += ["=" * 62, "SUMMARY", ""]
    lines.append("  By Difficulty:")
    for d, c in sorted(diff_counts.items()):
        lines.append(f"    {d:8s}: {c}")
    lines.append("")
    lines.append("  By Topic:")
    for t, c in sorted(topic_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {t}: {c}")
    lines += ["", "=" * 62]

    return "\n".join(lines)


def format_study_guide(questions: list[Question], title: str) -> str:
    """Combined exam with correct answers marked — useful as a study guide."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 62,
        f"{title.upper()}  —  STUDY GUIDE (WITH ANSWERS)",
        f"Generated : {ts}",
        f"Questions : {len(questions)}",
        "=" * 62,
        "",
    ]

    for q in questions:
        correct = q.get("correct_answer", "?")
        explanation = q.get("explanation", "")
        lines.append(
            f"Q{q['id']}.  [{q.get('difficulty', '')}]  {q.get('topic', '')}"
        )
        lines.append(q["question"])
        lines.append("")
        for letter in ("A", "B", "C", "D"):
            opt = q.get("options", {}).get(letter, "")
            if opt:
                marker = "✓" if letter == correct else " "
                lines.append(f"  {marker} {letter})  {opt}")
        lines.append("")
        if explanation:
            lines.append(f"  → {explanation}")
        lines.append("")

    return "\n".join(lines)


# ─── Score calculation ────────────────────────────────────────────────────────

def calculate_score(
    questions: list[Question],
    user_answers: dict[int, str],
) -> dict:
    """
    Compute quiz score and categorise results.

    Args:
        questions: List of question dicts.
        user_answers: Mapping of question id → chosen letter (A/B/C/D).

    Returns:
        Dict with keys: total, correct, percentage, by_difficulty, wrong_questions.
    """
    correct_count = 0
    by_difficulty: dict[str, dict[str, int]] = {}
    wrong: list[Question] = []

    for q in questions:
        qid = q["id"]
        diff = q.get("difficulty", "Medium").capitalize()
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}

        by_difficulty[diff]["total"] += 1

        if user_answers.get(qid) == q.get("correct_answer"):
            correct_count += 1
            by_difficulty[diff]["correct"] += 1
        else:
            wrong.append(q)

    total = len(questions)
    return {
        "total": total,
        "correct": correct_count,
        "percentage": round(correct_count / total * 100, 1) if total else 0.0,
        "by_difficulty": by_difficulty,
        "wrong_questions": wrong,
    }


def safe_filename(title: str) -> str:
    """Convert a title to a safe file name base."""
    import re
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"[\s]+", "_", safe.strip())
    return safe[:60]
