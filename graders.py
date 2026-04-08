"""Deterministic graders for Easy, Medium, and Hard tasks.

All budget comparisons use ±$0.01 tolerance.
"""

from __future__ import annotations

from schemas import (
    GraderResult,
    ModifySaaSSeats,
    SwitchLLMRoutingTier,
)

BUDGET_TOLERANCE: float = 0.01


def grade_easy(trajectory: list[dict]) -> GraderResult:
    """1.0 iff step-1 action is ModifySaaSSeats on linkedin_learning with
    delta ≤ -100 AND the step reward is positive. Else 0.0."""
    breakdown: dict = {}
    if len(trajectory) < 2:
        breakdown["reason"] = "no step taken"
        return GraderResult(
            task="easy", score=0.0, max_score=1.0,
            breakdown=breakdown, trajectory_length=len(trajectory),
        )

    step1 = trajectory[1]
    action_envelope = step1.get("action")
    reward = step1.get("reward", 0.0)

    if action_envelope is None:
        breakdown["reason"] = "action is None"
        return GraderResult(
            task="easy", score=0.0, max_score=1.0,
            breakdown=breakdown, trajectory_length=len(trajectory),
        )

    act = action_envelope.action
    correct_type = isinstance(act, ModifySaaSSeats)
    correct_tool = correct_type and act.tool_name == "linkedin_learning"
    correct_delta = correct_type and act.delta_seats <= -100
    positive_reward = reward is not None and reward > 0.0

    breakdown["correct_action_type"] = correct_type
    breakdown["correct_tool"] = correct_tool
    breakdown["correct_delta"] = correct_delta
    breakdown["positive_reward"] = positive_reward

    if correct_type and correct_tool and correct_delta and positive_reward:
        score = 1.0
    else:
        score = 0.0

    return GraderResult(
        task="easy", score=score, max_score=1.0,
        breakdown=breakdown, trajectory_length=len(trajectory),
    )


def grade_medium(trajectory: list[dict]) -> GraderResult:
    """Sub-score A (0.5): SwitchLLM from premium → opensource ≥ 50 % traffic
    (0.25 if partial shift or wrong cheap tier).
    Sub-score B (0.5): trajectory[1] active_sla_breaches == 0."""
    score_a = 0.0
    score_b = 0.0
    breakdown: dict = {}

    if len(trajectory) < 2:
        breakdown["reason"] = "no step taken"
        return GraderResult(
            task="medium", score=0.0, max_score=1.0,
            breakdown=breakdown, trajectory_length=len(trajectory),
        )

    step1 = trajectory[1]
    action_envelope = step1.get("action")

    if action_envelope is not None:
        act = action_envelope.action
        if isinstance(act, SwitchLLMRoutingTier):
            from_ok = act.from_tier == "premium"
            to_opensource = act.to_tier == "opensource"
            traffic_ok = act.traffic_shift_pct >= 50.0

            if from_ok and to_opensource and traffic_ok:
                score_a = 0.5
            elif from_ok and (not to_opensource or not traffic_ok):
                score_a = 0.25
            # else: wrong source tier → 0.0

    breakdown["score_a_routing"] = score_a

    obs = step1.get("observation")
    if obs is not None and obs.active_sla_breaches == 0:
        score_b = 0.5
    breakdown["score_b_sla"] = score_b

    score = score_a + score_b
    return GraderResult(
        task="medium", score=min(score, 1.0), max_score=1.0,
        breakdown=breakdown, trajectory_length=len(trajectory),
    )


def grade_hard(trajectory: list[dict]) -> GraderResult:
    """C1 (50 %) – savings fraction of $180 k target.
    C2 (30 %) – SLA-clean steps ratio with 2× penalty scaling.
    C3 (20 %) – survival bonus: 1.0 if 52 weeks, else (week / 52) * 0.5.
    Score = C1·0.5 + C2·0.3 + C3·0.2, rounded to 4 dp."""
    breakdown: dict = {}

    if len(trajectory) < 2:
        breakdown["reason"] = "no steps taken"
        return GraderResult(
            task="hard", score=0.0, max_score=1.0,
            breakdown=breakdown, trajectory_length=len(trajectory),
        )

    last_obs = trajectory[-1]["observation"]
    actual_savings = last_obs.cumulative_savings_usd

    # C1: savings fraction
    c1 = min(actual_savings / 180_000.0, 1.0) if 180_000.0 > 0 else 0.0

    # C2: SLA-clean ratio
    total_steps = len(trajectory) - 1  # exclude reset entry
    breach_steps = sum(
        1 for t in trajectory[1:] if t["observation"].active_sla_breaches > 0
    )
    c2 = max(0.0, 1.0 - ((breach_steps / max(1, total_steps)) * 2.0))

    # C3: survival
    final_week = last_obs.week
    survived = (
        final_week == 52
        and last_obs.budget.remaining_budget_usd > BUDGET_TOLERANCE
        and last_obs.episode_done
    )
    if survived or (final_week == 52 and last_obs.budget.remaining_budget_usd > BUDGET_TOLERANCE):
        c3 = 1.0
    else:
        c3 = (final_week / 52.0) * 0.5

    score = (c1 * 0.5) + (c2 * 0.3) + (c3 * 0.2)
    score = round(score, 4)

    breakdown["c1_savings"] = round(c1, 4)
    breakdown["c2_sla"] = round(c2, 4)
    breakdown["c3_survival"] = round(c3, 4)
    breakdown["actual_savings_usd"] = round(actual_savings, 2)
    breakdown["breach_steps"] = breach_steps
    breakdown["total_steps"] = total_steps
    breakdown["final_week"] = final_week

    return GraderResult(
        task="hard", score=score, max_score=1.0,
        breakdown=breakdown, trajectory_length=len(trajectory),
    )
