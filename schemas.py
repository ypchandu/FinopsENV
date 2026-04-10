from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class SaaSToolStats(BaseModel):
    tool_name: str
    total_seats: int
    active_seats: int
    inactive_seats: int
    cost_per_seat_usd: float
    monthly_cost_usd: float


class LLMTierStats(BaseModel):
    tier_name: str
    model_id: str
    requests_this_week: int
    cost_per_1k_tokens_usd: float
    weekly_spend_usd: float
    p95_latency_ms: float
    sla_latency_threshold_ms: float


class BudgetState(BaseModel):
    fiscal_week: int
    annual_budget_usd: float
    spent_to_date_usd: float
    remaining_budget_usd: float
    weekly_burn_rate_usd: float
    projected_overrun_usd: float


class Observation(BaseModel):
    week: int = Field(..., ge=1, le=52)
    budget: BudgetState
    saas_tools: list[SaaSToolStats]
    llm_tiers: list[LLMTierStats]
    active_sla_breaches: int
    cumulative_savings_usd: float
    episode_done: bool


class ModifySaaSSeats(BaseModel):
    action_type: Literal["ModifySaaSSeats"] = "ModifySaaSSeats"
    tool_name: str
    delta_seats: int
    justification: str


class SwitchLLMRoutingTier(BaseModel):
    action_type: Literal["SwitchLLMRoutingTier"] = "SwitchLLMRoutingTier"
    from_tier: str
    to_tier: str
    traffic_shift_pct: float = Field(..., ge=0.0, le=100.0)
    justification: str


class NoOp(BaseModel):
    action_type: Literal["NoOp"] = "NoOp"
    justification: str


class ActionEnvelope(BaseModel):
    task: Literal["easy", "medium", "hard"]
    action: Union[ModifySaaSSeats, SwitchLLMRoutingTier, NoOp] = Field(discriminator="action_type")


class RewardBreakdown(BaseModel):
    savings_reward: float
    sla_penalty: float
    churn_penalty: float
    invalid_action_penalty: float
    net_reward: float


class StepResult(BaseModel):
    observation: Observation
    reward: float
    reward_breakdown: RewardBreakdown
    done: bool
    info: dict


class GraderResult(BaseModel):
    task: Literal["easy", "medium", "hard"]
    score: float
    max_score: float = 1.0
    breakdown: dict
    trajectory_length: int

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Strictly enforce that the score is between 0 and 1 (exclusive)."""
        if not (0.0 < v < 1.0):
            # If the logic somehow failed, we clamp it here as a final safety net
            return max(0.010, min(0.990, v))
        return v
