"""
Experiment configuration — defines all runs for Experiment 1 (Baseline) and Experiment 2 (Model Comparison).
"""

# ---------------------------------------------------------------------------
# Mission objective (unchanged across all experiments)
# ---------------------------------------------------------------------------
OBJECTIVE = """\
Select the single best Course of Action (CoA) to prevent an imminent attack by the Terrmisous group on a commercial cargo ship carrying chemical agents near the civilian port AricikPortus.

Choose ONE option only (CoA 1, CoA 2, or CoA 3). Do not combine options.

CoA 1: Disable the VicikPortus power grid using previously implanted BlackEnergy3-based malware to prevent loading.
CoA 2: Temporarily neutralize the VicikPortus civilian pump station using a DDoS attack exploiting an unpatched vulnerability, preventing fueling.
CoA 3: Conduct GNSS spoofing to alter the cargo ship's position, velocity, and heading after departure so it cannot reach its intended target.
"""

# ---------------------------------------------------------------------------
# Model settings (applied to all experiment runs)
# ---------------------------------------------------------------------------
EXPERIMENT_SETTINGS = {
    "temperature": 0.2,
    "top_k": 30,
    "top_p": 0.95,
    "max_tokens": 800,
}

# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------
MODELS = {
    "zephyr":  "HuggingFaceH4/zephyr-7b-beta",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# ---------------------------------------------------------------------------
# Subagent YAML configs (used when use_subagents=True)
# ---------------------------------------------------------------------------
SUBAGENT_CONFIGS = [
    {
        "yaml_path": "agents/config/LegalAdvisor.yaml",
        "kb_path":   "agents/kb/LegalAdvisor",
        "tool_name": "ask_legal",
    },
    {
        "yaml_path": "agents/config/CyberOperationsExpert.yaml",
        "kb_path":   "agents/kb/CyberOperationsExpert",
        "tool_name": "ask_cyber_operations",
    },
    {
        "yaml_path": "agents/config/MilitaryOperationsExpert.yaml",
        "kb_path":   "agents/kb/MilitaryOperationsExpert",
        "tool_name": "ask_military_operations",
    },
]

# ---------------------------------------------------------------------------
# Experiment 1 — Baseline (Zephyr 7B only)
# ---------------------------------------------------------------------------
EXPERIMENT_1_RUNS = [
    {
        "experiment_id": "EXP1-001",
        "phase": 1,
        "model_key": "zephyr",
        "agent_config": "commander-only",
        "commander_yaml": "experiments/configs/Commander_solo.yaml",
        "use_subagents": False,
    },
    {
        "experiment_id": "EXP1-002",
        "phase": 1,
        "model_key": "zephyr",
        "agent_config": "commander-subagents",
        "commander_yaml": "agents/config/Commander.yaml",
        "use_subagents": True,
    },
]

# ---------------------------------------------------------------------------
# Experiment 2 — Model Comparison (3 models x 2 configurations)
# ---------------------------------------------------------------------------
EXPERIMENT_2_RUNS = []
_exp2_idx = 0
for _model_key in ["zephyr", "mistral", "llama"]:
    for _use_sub, _config_label, _cmd_yaml in [
        (False, "commander-only",      "experiments/configs/Commander_solo.yaml"),
        (True,  "commander-subagents", "agents/config/Commander.yaml"),
    ]:
        _exp2_idx += 1
        EXPERIMENT_2_RUNS.append({
            "experiment_id": f"EXP2-{_exp2_idx:03d}",
            "phase": 2,
            "model_key": _model_key,
            "agent_config": _config_label,
            "commander_yaml": _cmd_yaml,
            "use_subagents": _use_sub,
        })

# ---------------------------------------------------------------------------
# Combined list of all runs
# ---------------------------------------------------------------------------
ALL_RUNS = EXPERIMENT_1_RUNS + EXPERIMENT_2_RUNS
