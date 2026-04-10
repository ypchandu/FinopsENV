from __future__ import annotations
from schemas import GraderResult

BUDGET_TOLERANCE: float = 0.01
MIN_SCORE = 0.001
MAX_SCORE = 0.998

def grade_easy(trajectory: list[dict]) -> GraderResult:
    breakdown: dict = {}
    if len(trajectory) < 2:
        breakdown["reason"] = "no step taken"
        return GraderResult(task="easy", score=MIN_SCORE, max_score=1.0, breakdown=breakdown, trajectory_length=len(trajectory))

    step1 = trajectory[1]
    
    act = step1.get("action", {})
    if hasattr(act, "action"):
        act = act.action
    elif isinstance(act, dict) and "action" in act:
        act = act["action"]

    reward = step1.get("reward", 0.0)

    if not act:
        breakdown["reason"] = "action is None"
        return GraderResult(task="easy", score=MIN_SCORE, max_score=1.0, breakdown=breakdown, trajectory_length=len(trajectory))

    act_dict = act if isinstance(act, dict) else vars(act)

    # FIXED: Indentation restored here
    correct_type = act_dict.get("action_type") == "ModifySaaSSeats"
    correct_tool = correct_type and act_dict.get("tool_name") == "linkedin_learning"
    correct_delta = correct_type and int(act_dict.get("delta_seats", 0)) <= -100
    positive_reward = reward is not None and float(reward) > 0.0

    breakdown["correct_action_type"] = correct_type
    breakdown["correct_tool"] = correct_tool
    breakdown["correct_delta"] = correct_delta
    breakdown["positive_reward"] = positive_reward

    score = MAX_SCORE if (correct_type and correct_tool and correct_delta and positive_reward) else MIN_SCORE

    return GraderResult(
        task="easy", score=score, max_score=1.0,
        breakdown=breakdown, trajectory_length=len(trajectory)
    )


def grade_medium(trajectory: list[dict]) -> GraderResult:
    score_a = 0.0
    score_b = 0.0
    breakdown: dict = {}

    if len(trajectory) < 2:
        breakdown["reason"] = "no step taken"
        return GraderResult(task="medium", score=MIN_SCORE, max_score=1.0, breakdown=breakdown, trajectory_length=len(trajectory))

    step1 = trajectory[1]
    
    act = step1.get("action", {})
    if hasattr(act, "action"):
        act = act.action
    elif isinstance(act, dict) and "action" in act:
        act = act["action"]

    act_dict = act if isinstance(act, dict) else vars(act)

    # FIXED: Indentation restored here
    if act_dict and act_dict.get("action_type") == "SwitchLLMRoutingTier":
        from_ok = act_dict.get("from_tier") == "premium"
        to_opensource = act_dict.get("to_tier") == "opensource"
        traffic_ok = float(act_dict.get("traffic_shift_pct", 0)) >= 50.0

        if from_ok and to_opensource and traffic_ok:
            score_a = 0.5
        elif from_ok and (not to_opensource or not traffic_ok):
            score_a = 0.25

    breakdown["score_a_routing"] = score_a

    obs = step1.get("observation", {})
    obs_dict = obs if isinstance(obs, dict) else vars(obs)
    
    if obs_dict and obs_dict.get("active_sla_breaches") == 0:
        score_b = 0.5
    breakdown["score_b_sla"] = score_b

    raw_score = score_a + score_b
    final_score = max(MIN_SCORE, min(MAX_SCORE, raw_score))

    return GraderResult(
        task="medium", score=final_score, max_score=1.0,
        breakdown=breakdown, trajectory_length=len(trajectory)
    )


def grade_hard(trajectory: list[dict]) -> GraderResult:
    """C1 (50 %) – savings fraction of $180 k target.
    C2 (30 %) – SLA-clean steps ratio with 2× penalty scaling.
    C3 (20 %) – survival bonus: 1.0 if 52 weeks, else (week / 52) * 0.5.
    Score = C1·0.5 + C2·0.3 + C3·0.2, rounded to 4 dp."""
    breakdown: dict = {}

    if len(trajectory) < 2:
        breakdown["reason"] = "no steps taken"
        return GraderResult(task="hard", score=MIN_SCORE, max_score=1.0, breakdown=breakdown, trajectory_length=len(trajectory))

    # Resilient obs extraction
    last_step = trajectory[-1]
    last_obs = last_step.get("observation", {})
    last_obs_dict = last_obs if isinstance(last_obs, dict) else vars(last_obs)

    actual_savings = float(last_obs_dict.get("cumulative_savings_usd", 0.0))

    # C1: savings fraction
    c1 = min(actual_savings / 180_000.0, 1.0) if 180_000.0 > 0 else 0.0

    # C2: SLA-clean ratio
    total_steps = len(trajectory) - 1  # exclude reset entry
    breach_steps = 0
    for t in trajectory[1:]:
        step_obs = t.get("observation", {})
        step_obs_dict = step_obs if isinstance(step_obs, dict) else vars(step_obs)
        if int(step_obs_dict.get("active_sla_breaches", 0)) > 0:
            breach_steps += 1
            
    c2 = max(0.0, 1.0 - ((breach_steps / max(1, total_steps)) * 2.0))

    # C3: survival
    final_week = int(last_obs_dict.get("week", 1))
    
    # Nested budget check
    budget = last_obs_dict.get("budget", {})
    budget_dict = budget if isinstance(budget, dict) else vars(budget)
    rem_budget = float(budget_dict.get("remaining_budget_usd", 0.0))
    episode_done = bool(last_obs_dict.get("episode_done", False))

    survived = (final_week == 52 and rem_budget > BUDGET_TOLERANCE and episode_done)
    
    if survived or (final_week == 52 and rem_budget > BUDGET_TOLERANCE):
        c3 = 1.0
    else:
        c3 = (final_week / 52.0) * 0.5

    score = (c1 * 0.5) + (c2 * 0.3) + (c3 * 0.2)
    score = round(score, 4)
    final_score = max(MIN_SCORE, min(MAX_SCORE, score))

    breakdown["c1_savings"] = round(c1, 4)
    breakdown["c2_sla"] = round(c2, 4)
    breakdown["c3_survival"] = round(c3, 4)
    breakdown["actual_savings_usd"] = round(actual_savings, 2)
    breakdown["breach_steps"] = breach_steps
    breakdown["total_steps"] = total_steps
    breakdown["final_week"] = final_week

    return GraderResult(
        task="hard", score=final_score, max_score=1.0,
        breakdown=breakdown, trajectory_length=len(trajectory)
    )
