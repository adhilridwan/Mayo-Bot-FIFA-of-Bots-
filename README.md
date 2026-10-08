# Mayo Bot by Team Shawarma

**Mayo FTW!!!**:


> **Architecture Summary**: Mayo is an adversarial, deterministic search agent built on a **1.5-ply forward minimax engine** with decoupled action selection and closed-loop trajectory simulation. Operating in pure Python with zero external dependencies, Mayo evaluates a **216-action combinatorial space** (9 movement vectors $\times$ 24 kick trajectories) per decision cycle, maintaining an average inference latency of **~13 ms** (against the 2,000 ms competition ceiling).


### **System Specifications**

| Component | Specification |
|---|---|
| **Paradigm** | Deterministic Minimax Lookahead & Heuristic Planning |
| **Dependencies** | Python 3.11+ Standard Library (Zero `pip` dependencies) |
| **Action Space** | 216 discrete actions per iteration |
| **Decision Latency** | ~13 ms mean, < 300 ms p99 (Limit: 2,000 ms) |
| **Memory Footprint** | Pure Python memory model (< 15 MB) |
| **Safety Standard** | Passes official static security checker with 0 errors |






# Execution

```powershell
python -m team_bot.bot --model team_bot/models/policy.json
```
