"""FinOps RL Environment – Phase 2 Core Logic.

Implements FinOpsEnv with reset / step / state and full trajectory tracking.
All math follows the locked blueprint exactly, with the weekly-savings time-scale
bug fix applied (÷ 4.33).
"""

from __future__ import annotations

import copy
import random
import time
from typing import Optional

from schemas import (
    ActionEnvelope,
    BudgetState,
    LLMTierStats,
    ModifySaaSSeats,
    NoOp,
    Observation,
    RewardBreakdown,
    SaaSToolStats,
    StepResult,
    SwitchLLMRoutingTier,
)

# ── Reward constants ────────────────────────────────────────────────────────
SAVINGS_WEIGHT: float = 0.01
SLA_PENALTY: float = -75.0
CHURN_PENALTY: float = -150.0
INVALID_PENALTY: float = -50.0
BANKRUPTCY_PENALTY: float = -500.0
IDLE_PENALTY: float = -5.0

# ── Operational constants ───────────────────────────────────────────────────
OVERHEAD_WEEKLY: float = 1_200.0
MONTHLY_TO_WEEKLY: float = 4.33
SLA_THRESHOLD_MS: float = 800.0
BUDGET_TOLERANCE: float = 0.01


class FinOpsEnv:
    """OpenEnv-compliant FinOps simulation environment."""

    def __init__(self) -> None:
        self.task: Optional[str] = None
        self.week: int = 1
        self.saas_tools: list[SaaSToolStats] = []
        self.llm_tiers: list[LLMTierStats] = []
        self.budget: Optional[BudgetState] = None
        self.active_sla_breaches: int = 0
        self.cumulative_savings: float = 0.0
        self.episode_done: bool = False
        self.trajectory: list[dict] = []
        self._rng: Optional[random.Random] = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _compute_weekly_burn(self) -> float:
        saas = sum(t.monthly_cost_usd / MONTHLY_TO_WEEKLY for t in self.saas_tools)
        llm = sum(t.weekly_spend_usd for t in self.llm_tiers)
        return saas + llm + OVERHEAD_WEEKLY

    def _build_observation(self) -> Observation:
        burn = self._compute_weekly_burn()
        remaining_weeks = max(1, 52 - self.week + 1)
        remaining = self.budget.annual_budget_usd - self.budget.spent_to_date_usd
        projected = max(
            0.0,
            self.budget.spent_to_date_usd + burn * remaining_weeks - self.budget.annual_budget_usd,
        )
        self.budget.fiscal_week = self.week
        self.budget.weekly_burn_rate_usd = round(burn, 2)
        self.budget.remaining_budget_usd = round(remaining, 2)
        self.budget.projected_overrun_usd = round(projected, 2)

        return Observation(
            week=self.week,
            budget=self.budget.model_copy(),
            saas_tools=[t.model_copy() for t in self.saas_tools],
            llm_tiers=[t.model_copy() for t in self.llm_tiers],
            active_sla_breaches=self.active_sla_breaches,
            cumulative_savings_usd=round(self.cumulative_savings, 2),
            episode_done=self.episode_done,
        )

    def _count_sla_breaches(self) -> int:
        return sum(1 for t in self.llm_tiers if t.p95_latency_ms > t.sla_latency_threshold_ms)

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task: str) -> Observation:
        self.task = task
        self.week = 1
        self.cumulative_savings = 0.0
        self.active_sla_breaches = 0
        self.episode_done = False
        self.trajectory = []
        self._rng = None

        if task == "easy":
            self._init_easy()
        elif task == "medium":
            self._init_medium()
        elif task == "hard":
            self._init_hard()
        else:
            raise ValueError(f"Unknown task: {task}")

        obs = self._build_observation()
        self.trajectory.append(
            {"step": 0, "action": None, "observation": obs, "reward": None, "reward_breakdown": None}
        )
        return obs

    # ── deterministic starting states ────────────────────────────────────────

    def _init_easy(self) -> None:
        self.budget = BudgetState(
            fiscal_week=1,
            annual_budget_usd=500_000.0,
            spent_to_date_usd=0.0,
            remaining_budget_usd=500_000.0,
            weekly_burn_rate_usd=0.0,
            projected_overrun_usd=0.0,
        )
        self.saas_tools = [
            SaaSToolStats(
                tool_name="linkedin_learning", total_seats=150, active_seats=50,
                inactive_seats=100, cost_per_seat_usd=30.0, monthly_cost_usd=4_500.0,
            ),
            SaaSToolStats(
                tool_name="slack", total_seats=200, active_seats=150,
                inactive_seats=50, cost_per_seat_usd=12.50, monthly_cost_usd=2_500.0,
            ),
            SaaSToolStats(
                tool_name="salesforce", total_seats=80, active_seats=80,
                inactive_seats=0, cost_per_seat_usd=75.0, monthly_cost_usd=6_000.0,
            ),
        ]
        self.llm_tiers = []

    def _init_medium(self) -> None:
        self.budget = BudgetState(
            fiscal_week=1,
            annual_budget_usd=200_000.0,
            spent_to_date_usd=180_000.0,
            remaining_budget_usd=20_000.0,
            weekly_burn_rate_usd=0.0,
            projected_overrun_usd=0.0,
        )
        self.saas_tools = []
        self.llm_tiers = [
            LLMTierStats(
                tier_name="premium", model_id="claude-opus-4",
                requests_this_week=50_000, cost_per_1k_tokens_usd=0.075,
                weekly_spend_usd=5_200.0, p95_latency_ms=420.0,
                sla_latency_threshold_ms=800.0,
            ),
            LLMTierStats(
                tier_name="standard", model_id="gpt-4o-mini",
                requests_this_week=30_000, cost_per_1k_tokens_usd=0.002,
                weekly_spend_usd=600.0, p95_latency_ms=210.0,
                sla_latency_threshold_ms=800.0,
            ),
            LLMTierStats(
                tier_name="opensource", model_id="llama-3-70b",
                requests_this_week=10_000, cost_per_1k_tokens_usd=0.0004,
                weekly_spend_usd=80.0, p95_latency_ms=680.0,
                sla_latency_threshold_ms=800.0,
            ),
        ]

    def _init_hard(self) -> None:
        self._rng = random.Random(time.time())
        self.budget = BudgetState(
            fiscal_week=1,
            annual_budget_usd=1_200_000.0,
            spent_to_date_usd=0.0,
            remaining_budget_usd=1_200_000.0,
            weekly_burn_rate_usd=0.0,
            projected_overrun_usd=0.0,
        )
        self.saas_tools = [
            SaaSToolStats(
                tool_name="salesforce", total_seats=300, active_seats=100,
                inactive_seats=200, cost_per_seat_usd=75.0, monthly_cost_usd=22_500.0,
            ),
            SaaSToolStats(
                tool_name="linkedin_learning", total_seats=1000, active_seats=500,
                inactive_seats=500, cost_per_seat_usd=30.0, monthly_cost_usd=500_000.0,
            ),
            SaaSToolStats(
                tool_name="slack", total_seats=1_000, active_seats=420,
                inactive_seats=580, cost_per_seat_usd=12.50, monthly_cost_usd=12_500.0,
            ),
            SaaSToolStats(
                tool_name="zoom", total_seats=400, active_seats=190,
                inactive_seats=210, cost_per_seat_usd=20.0, monthly_cost_usd=8_000.0,
            ),
            SaaSToolStats(
                tool_name="github", total_seats=250, active_seats=148,
                inactive_seats=102, cost_per_seat_usd=21.0, monthly_cost_usd=5_250.0,
            ),
        ]
        self.llm_tiers = [
            LLMTierStats(
                tier_name="premium", model_id="claude-opus-4",
                requests_this_week=80_000, cost_per_1k_tokens_usd=0.575,
                weekly_spend_usd=8_320.0, p95_latency_ms=430.0,
                sla_latency_threshold_ms=1000.0,
            ),
            LLMTierStats(
                tier_name="standard", model_id="gpt-4o-mini",
                requests_this_week=60_000, cost_per_1k_tokens_usd=0.042,
                weekly_spend_usd=1_200.0, p95_latency_ms=215.0,
                sla_latency_threshold_ms=1000.0,
            ),
            LLMTierStats(
                tier_name="opensource", model_id="llama-3-70b",
                requests_this_week=20_000, cost_per_1k_tokens_usd=0.0004,
                weekly_spend_usd=160.0, p95_latency_ms=690.0,
                sla_latency_threshold_ms=800.0,
            ),
        ]

    # ── hard-mode perturbation (deterministic via seeded RNG) ────────────────

    def _apply_perturbation(self) -> None:
        if self.task != "hard" or self._rng is None:
            return
        roll = self._rng.random()
        if roll < 0.30:
            tool = self._rng.choice(self.saas_tools)
            n = self._rng.randint(5, 25)
            tool.total_seats += n
            tool.inactive_seats += n
            tool.monthly_cost_usd = tool.total_seats * tool.cost_per_seat_usd
        elif roll < 0.50:
            for tier in self.llm_tiers:
                if tier.tier_name == "premium":
                    old_reqs = tier.requests_this_week
                    tier.requests_this_week += 10_000
                    if old_reqs > 0:
                        tier.weekly_spend_usd *= tier.requests_this_week / old_reqs
                    break
        elif roll < 0.60:
            per_tool = 25 // len(self.saas_tools)
            remainder = 25 % len(self.saas_tools)
            for i, tool in enumerate(self.saas_tools):
                add = per_tool + (1 if i < remainder else 0)
                tool.total_seats += add
                tool.inactive_seats += add
                tool.monthly_cost_usd = tool.total_seats * tool.cost_per_seat_usd
        # else (≥0.60): no event – 40 %

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: ActionEnvelope) -> StepResult:
        if self.episode_done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        # 1. Hard-mode perturbation at TOP of step
        self._apply_perturbation()

        # 2. Snapshot pre-action burn
        pre_burn = self._compute_weekly_burn()

        # 3. Process action
        weekly_savings = 0.0
        sla_breaches = 0
        churn_events = 0
        invalid = False

        act = action.action
        if isinstance(act, ModifySaaSSeats):
            r = self._handle_modify_saas(act)
            weekly_savings = r["savings"]
            churn_events = r["churn_events"]
            invalid = r["invalid"]
        elif isinstance(act, SwitchLLMRoutingTier):
            r = self._handle_switch_llm(act)
            sla_breaches = r["sla_breaches"]
            invalid = r["invalid"]
        # NoOp → nothing

        # 4. Clock: deduct weekly burn (pre-action) then credit savings
        self.budget.spent_to_date_usd += pre_burn
        self.budget.spent_to_date_usd -= weekly_savings
        self.cumulative_savings += weekly_savings
        self.budget.remaining_budget_usd = (
            self.budget.annual_budget_usd - self.budget.spent_to_date_usd
        )

        # 5. Advance week
        self.week += 1
        bankrupt = self.budget.remaining_budget_usd <= BUDGET_TOLERANCE
        if bankrupt or self.week > 52:
            self.week = min(self.week, 52)
            self.episode_done = True

        # 6. Recount SLA breaches for observation
        self.active_sla_breaches = self._count_sla_breaches()

        # 7. Reward
        rb = self._compute_reward(
            weekly_savings=weekly_savings,
            sla_breach_count=sla_breaches,
            churn_count=churn_events,
            invalid=invalid,
            bankrupt=bankrupt,
            is_noop=isinstance(act, NoOp),
        )

        obs = self._build_observation()
        self.trajectory.append(
            {
                "step": len(self.trajectory),
                "action": action,
                "observation": obs,
                "reward": rb.net_reward,
                "reward_breakdown": rb,
            }
        )
        return StepResult(
            observation=obs,
            reward=rb.net_reward,
            reward_breakdown=rb,
            done=self.episode_done,
            info={"week": self.week, "cumulative_savings": round(self.cumulative_savings, 2)},
        )

    # ── action handlers ──────────────────────────────────────────────────────

    def _handle_modify_saas(self, act: ModifySaaSSeats) -> dict:
        tool = next((t for t in self.saas_tools if t.tool_name == act.tool_name), None)
        if tool is None:
            return {"savings": 0.0, "churn_events": 0, "invalid": True}

        new_total = tool.total_seats + act.delta_seats
        churn = 0
        if new_total < tool.active_seats:
            churn = tool.active_seats - new_total
            new_total = tool.active_seats
        if new_total < 0:
            new_total = 0

        seats_removed = max(0, tool.total_seats - new_total)
        # BUG-FIX: monthly cost ÷ 4.33 to convert to weekly
        weekly_savings = (seats_removed * tool.cost_per_seat_usd) / MONTHLY_TO_WEEKLY

        tool.total_seats = new_total
        tool.inactive_seats = max(0, tool.total_seats - tool.active_seats)
        tool.monthly_cost_usd = tool.total_seats * tool.cost_per_seat_usd

        return {"savings": weekly_savings, "churn_events": 1 if churn > 0 else 0, "invalid": False}

    def _handle_switch_llm(self, act: SwitchLLMRoutingTier) -> dict:
        from_tier = next((t for t in self.llm_tiers if t.tier_name == act.from_tier), None)
        to_tier = next((t for t in self.llm_tiers if t.tier_name == act.to_tier), None)
        if from_tier is None or to_tier is None or from_tier.tier_name == to_tier.tier_name:
            return {"sla_breaches": 0, "invalid": True}

        requests_to_move = int(from_tier.requests_this_week * (act.traffic_shift_pct / 100.0))
        old_from_reqs = from_tier.requests_this_week
        old_to_reqs = to_tier.requests_this_week

        from_tier.requests_this_week -= requests_to_move
        to_tier.requests_this_week += requests_to_move

        # Proportional spend update
        if old_from_reqs > 0:
            from_tier.weekly_spend_usd *= from_tier.requests_this_week / old_from_reqs
        else:
            from_tier.weekly_spend_usd = 0.0
        if old_to_reqs > 0:
            to_tier.weekly_spend_usd *= to_tier.requests_this_week / old_to_reqs
        elif to_tier.requests_this_week > 0:
            to_tier.weekly_spend_usd = (
                to_tier.requests_this_week * to_tier.cost_per_1k_tokens_usd / 1_000.0
            )

        # Latency pressure on destination tier
        new_to_reqs = to_tier.requests_this_week
        if new_to_reqs > 0 and requests_to_move > 0:
            latency_pressure_factor = 1.0 + (requests_to_move / new_to_reqs) * 0.4
            to_tier.p95_latency_ms *= latency_pressure_factor

        sla_breaches = self._count_sla_breaches()
        self.active_sla_breaches = sla_breaches
        return {"sla_breaches": sla_breaches, "invalid": False}

    # ── reward ───────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        weekly_savings: float,
        sla_breach_count: int,
        churn_count: int,
        invalid: bool,
        bankrupt: bool,
        is_noop: bool,
    ) -> RewardBreakdown:
        sav = weekly_savings * SAVINGS_WEIGHT
        sla = sla_breach_count * SLA_PENALTY
        churn = churn_count * CHURN_PENALTY
        inv = INVALID_PENALTY if invalid else 0.0
        bank = BANKRUPTCY_PENALTY if bankrupt else 0.0
        idle = IDLE_PENALTY if is_noop else 0.0
        net = sav + sla + churn + inv + bank + idle
        return RewardBreakdown(
            savings_reward=round(sav, 4),
            sla_penalty=round(sla + bank + idle, 4),
            churn_penalty=round(churn, 4),
            invalid_action_penalty=round(inv, 4),
            net_reward=round(net, 4),
        )

    # ── state accessor ───────────────────────────────────────────────────────

    def state(self) -> Observation:
        return self._build_observation()
