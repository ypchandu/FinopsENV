---
title: Syntax Squad FinopsENV
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 💸 Autonomous FinOps Agent

**An OpenEnv-compliant Reinforcement Learning environment simulating an enterprise Financial Operations (FinOps) system.**

*Submission for the Meta × Scaler PyTorch Hackathon (Round 1)* **Team:** Syntax Squad / OMNIP0TENT

---

## 🌎 1. The Vision: The Enterprise Waste Epidemic

Modern enterprise companies bleed capital through invisible software waste. Hundreds of thousands of dollars are lost annually to two primary culprits:
1. **SaaS Bloat:** Paying for unused "zombie" seats on platforms like Salesforce or LinkedIn Learning.
2. **AI Inference Inefficiency:** Sending simple, low-complexity queries to expensive, frontier LLMs when cheaper, faster open-source models would suffice.

The **Autonomous FinOps Agent** environment acts as a training ground for an intelligent, automated CFO. Operating on a simulated 52-week fiscal cadence, the AI agent must digest complex telemetry regarding seat utilization and API expenditure, and autonomously prune waste. If it cuts too aggressively, it triggers employee churn or SLA latency breaches. If it doesn't cut enough, the company goes bankrupt.

---

## 🏗️ 2. System Architecture

To ensure strict compliance with the OpenEnv 1.0.0 specification and maintain production-grade reliability, the project is structured across several distinct engineering layers.

### High-Level Design (HLD)
The system operates on a **Client-Server RL Loop**:
* **The Body (Environment Server):** A containerized, deterministic state machine hosted in the cloud via Hugging Face Spaces. It exposes standard REST endpoints (`/reset`, `/step`, `/state`).
* **The Brain (Inference Client):** A local, lightweight Python script (`inference.py`) that acts as the player. It fetches the state, queries a frontier model (`gpt-4o-mini`), and sends the action back to the cloud.

### Low-Level Design (LLD)
* **Transport Layer:** FastAPI powered by Uvicorn, wrapped in a Docker container for instant portability.
* **Validation Layer:** Strict Pydantic models. We utilize a **Discriminated Union** (`ActionEnvelope`) for the action space, mathematically preventing the LLM from hallucinating invalid API calls.
* **Simulation Engine:** A purely deterministic Python class (`FinOpsEnv`). Stochastic elements (like random traffic spikes or sudden headcount growth in the 'Hard' task) are controlled via seeded RNG (`random.seed(42)`) to ensure 100% reproducible grading.
* **Evaluation Layer:** Three distinct, deterministic grader functions (Easy, Medium, Hard) evaluate the final trajectory based on financial efficiency, SLA integrity, and survival.

---

## ⚙️ 3. The Reinforcement Learning Mechanics

### Observation Space (State)
At each step, the environment emits a rich JSON payload detailing the company's health:
* **Time:** Current week (1-52).
* **Finances:** Annual budget, spent-to-date, remaining runway, and weekly burn rate.
* **SaaS Telemetry:** List of software tools, active vs. inactive seats, and cost per seat.
* **LLM Telemetry:** Traffic volume, cost per 1k tokens, p95 latency, and SLA thresholds.

### Action Space
The agent can execute one of three strictly typed actions per week:
1. `ModifySaaSSeats`: Prune inactive licenses to save capital.
2. `SwitchLLMRoutingTier`: Shift traffic percentages between Premium, Standard, and Open-Source AI tiers.
3. `NoOp`: Take no action.

### The Reward Function
We avoided "sparse rewards." The environment provides a continuous gradient signal:
* **Positive:** +0.01 points per dollar saved.
* **Negative Constraints:** -75.0 points for breaking latency SLAs, -150.0 points for causing employee churn (firing active seats), and -500.0 points for bankruptcy.

---

## 🚀 4. How to Verify and Use the Environment

This environment is deployed live and is fully interactive. Judges and reviewers can verify the system through three different methods:

### Method A: The Interactive UI (For Humans)
We have mapped the root URL to an interactive Swagger UI dashboard containing a "How to Use" guide. 
1. Visit the live environment: **[https://omnip0tent-syntax-squad-finopsenv.hf.space](https://omnip0tent-syntax-squad-finopsenv.hf.space)**
2. You can manually act as the agent by clicking `POST /reset` to start a game, and `POST /step` to submit an action payload.

### Method B: The Automated Validator (For CI/CD)
To prove 100% compliance with the hackathon's OpenEnv specification, run the official validation tool against our cloud infrastructure:
```bash
openenv validate --url https://omnip0tent-syntax-squad-finopsenv.hf.space
```
Expect a `passed: true` JSON response confirming all 6 OpenEnv criteria.

### Method C: Running the Agent (For Inference)
To watch the AI solve the environment live on your machine:

1. Clone this repository.
2. Export your OpenAI API key: 
```bash
export OPENAI_API_KEY="sk-..."
```
3. Run the hard task:
```bash
python inference.py hard
```
*(This will generate the strictly formatted `[START]`, `[STEP]`, and `[END]` logs to stdout).*

---

## 📂 5. Project Structure
```text
├── app.py               # FastAPI server, endpoints, and UI redirect
├── environment.py       # Core state machine and simulation logic
├── schemas.py           # Pydantic models (Action, Observation, Validation)
├── graders.py           # Deterministic scoring logic for Easy/Med/Hard
├── inference.py         # LLM client script to solve the environment
├── Dockerfile           # Containerization configuration
├── requirements.txt     # Python dependencies
├── openenv.yaml         # Official OpenEnv metadata declaration
└── README.md            # Project documentation
```

## 📄 6. License
This project is licensed under the MIT License - see the LICENSE file for details.
Built for educational and benchmarking purposes during the Meta × Scaler Hackathon 2026.
