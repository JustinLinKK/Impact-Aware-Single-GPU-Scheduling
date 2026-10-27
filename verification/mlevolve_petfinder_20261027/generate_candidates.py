#!/usr/bin/env python3
"""Generate three genuine MLEvolve PetFinder drafts; never execute candidate code."""
from __future__ import annotations
import hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

# This process and all children cannot see a GPU. Codex CLI itself uses read-only sandbox.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MLEVOLVE = Path("/home/justin/MLEvolve")
OUT = Path(__file__).resolve().parent
DATA = MLEVOLVE / ".cache/petfinder-pawpularity-score"
sys.path.insert(0, str(MLEVOLVE))
os.chdir(MLEVOLVE)

from omegaconf import OmegaConf
from config import prep_cfg, load_task_desc, prep_agent_workspace
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from agents import draft_agent
import agents.coder.base_coder as base_coder
from utils.seed import set_global_seed
from utils.pipeline_logging import PipelineActionLogger
import llm.codex_cli as codex_cli

GOAL = """Build a simple PetFinder Pawpularity image-plus-tabular regression candidate using only the local `input/` dataset. `train.csv` has 12 numeric feature columns plus the Pawpularity target; images are in `train/` and test images in `test/`. The current workspace contains the real Kaggle files. Do not download data or use external services.

This run is a scheduler workload fixture. Produce a complete runnable Python script, but do not execute it. Use a deterministic 90/10 train/validation split (seed 42), 256x256 RGB image inputs and all 12 tabular inputs, and a small trainable CNN whose image embedding is fused with a small tabular MLP to predict one float Pawpularity value per image. Use a simple architecture (no pretrained weights, no ensemble, no network downloads), FP32 throughout, AdamW with lr=0.001 and weight_decay=0.0001, MSE loss with FP32 reduction, batch size 4, exactly 3 epochs and an explicit MAX_STEPS_PER_EPOCH=8 limit so the script has a bounded, inspectable update budget. Vary the small CNN/MLP widths or depth from draft to draft while retaining this contract.

Define a no-argument `CandidateAdapter` with these real methods: `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`. `build_model` must accept context. Both batch builders must honor `scenario['batch_size']`, use `scenario['fixture']` when supplied, and return mappings with exactly the model inputs `image` float32 [B,3,256,256], `tabular` float32 [B,12], and `target` float32 [B]. These adapter fixtures are synthetic CPU capture inputs, not training observations. The training and validation steps must call the same real model and use its actual MSE training loss; keep target and loss reductions FP32. Put all data loading, training, validation, checkpoint, prediction and submission side effects behind `if __name__ == '__main__':`. The runnable script must print `Final Validation Score` followed by validation RMSE. Keep model and dataset handling CPU-safe and import-safe."""

# Start from the checked-in sanitized defaults, then disable unrelated services and
# route every MLEvolve agent request through the logged-in Codex CLI provider.
cfg = OmegaConf.load(MLEVOLVE / "config.example.yaml")
exp = f"mlevolve_petfinder_codeonly_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
updates = {
    "data_dir": str(DATA), "dataset_dir": "", "desc_file": None, "goal": GOAL,
    "eval": "Minimize held-out validation RMSE for Pawpularity; lower is better.",
    "exp_name": exp, "exp_id": "", "log_dir": str(OUT / "logs"),
    "workspace_dir": str(OUT / "workspace"), "preprocess_data": False,
    "copy_data": False, "use_grading_server": False,
    "experiment": {"mode": "hardware_aware"},
    "scheduler": {"enabled": False},
    "hardware_knowledge": {"enabled": False},
    "lesson_profiles": {"enabled": False, "read_enabled": False, "write_enabled": False},
    "context_cache": {"enabled": False},
    "preflight": {"enabled": False, "enabled_modes": ["hardware_aware"], "target_profile": "cpu"},
    "agent": {
        "steps": 3, "initial_drafts": 3, "seed": 42, "time_limit": 3600,
        "hardware_context_enabled": False, "use_global_memory": False,
        "code": {"provider": "codex_cli", "model": "gpt-6-astra", "api_key": "", "base_url": ""},
        "feedback": {"provider": "codex_cli", "model": "gpt-6-astra", "api_key": "", "base_url": ""},
        "search": {"num_drafts": 3},
        "cuda_docs": {"enabled": False, "rollout_mode": "off"},
        "review": {"enabled": False},
    },
}
cfg = OmegaConf.merge(cfg, updates)
cfg = prep_cfg(cfg)
set_global_seed(cfg.agent.seed)
prep_agent_workspace(cfg)
task_desc = load_task_desc(cfg)
run_log = cfg.log_dir
run_log.mkdir(parents=True, exist_ok=True)
logger = PipelineActionLogger(run_log / "pipeline.sqlite3", run_id=cfg.exp_name, mode=cfg.experiment.mode)

# Config receipt is useful for replay and omits all credential-valued fields.
def redact(x):
    if isinstance(x, dict):
        return {k: ("[REDACTED]" if any(t in k.lower() for t in ("api_key", "token", "password", "secret")) else redact(v)) for k, v in x.items()}
    if isinstance(x, list): return [redact(v) for v in x]
    return str(x) if isinstance(x, Path) else x
(OUT / "effective_config.redacted.json").write_text(json.dumps(redact(OmegaConf.to_container(cfg, resolve=True)), indent=2, sort_keys=True, default=str) + "\n")

request_log = OUT / "codex_requests.redacted.jsonl"
real_codex_run = codex_cli._run

def logged_codex_run(prompt, *, model, json_schema, timeout_seconds):
    entry = {"started_utc": datetime.now(timezone.utc).isoformat(), "model": model,
             "provider": "codex_cli", "prompt_chars": len(prompt),
             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
             "structured_schema": json_schema is not None, "timeout_seconds": timeout_seconds}
    t0 = time.monotonic()
    try:
        result = real_codex_run(prompt, model=model, json_schema=json_schema, timeout_seconds=timeout_seconds)
        entry.update({"ok": True, "elapsed_seconds": round(time.monotonic()-t0, 3),
                      "response_sha256": hashlib.sha256(result[0].encode()).hexdigest()})
        return result
    except Exception as exc:
        entry.update({"ok": False, "elapsed_seconds": round(time.monotonic()-t0, 3),
                      "error_type": type(exc).__name__, "error": str(exc)[:500]})
        raise
    finally:
        with request_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

codex_cli._run = logged_codex_run

# MLEvolve `llm.query` supports Codex, but `llm.generate` currently falls
# through to Gemini for provider=codex_cli. Route all generation imports in this
# harness to the existing Codex CLI backend; do not alter production code.
def codex_generate_router(prompt, cfg, temperature=None, max_tokens=None, stop_tokens=None, json_schema=None, **kwargs):
    return codex_cli.generate(prompt=prompt, cfg=cfg, temperature=temperature, max_tokens=max_tokens, stop_tokens=stop_tokens, json_schema=json_schema, **kwargs)
import llm
original_llm_generate = llm.generate
llm.generate = codex_generate_router
for _module in list(sys.modules.values()):
    if getattr(_module, "generate", None) is original_llm_generate:
        setattr(_module, "generate", codex_generate_router)
base_coder.generate = codex_generate_router

raw_dir, reviewed_dir, prompts_dir = (OUT / x for x in ("raw", "reviewed", "prompts"))
raw_dir.mkdir(exist_ok=True); reviewed_dir.mkdir(exist_ok=True); prompts_dir.mkdir(exist_ok=True)
real_generate = draft_agent.plan_and_code_query
counter = {"n": len(list(raw_dir.glob("candidate_*.py")))}

def capture_generation(agent, prompt):
    plan, code = real_generate(agent, prompt)
    i = counter["n"]
    counter["n"] += 1
    (raw_dir / f"candidate_{i:02d}.py").write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
    (raw_dir / f"candidate_{i:02d}.plan.txt").write_text(plan + "\n", encoding="utf-8")
    ptxt = agent._serialize_prompt(prompt) or ""
    (prompts_dir / f"candidate_{i:02d}.prompt.md").write_text(ptxt, encoding="utf-8")
    return plan, code

draft_agent.plan_and_code_query = capture_generation
agent = AgentSearch(task_desc=task_desc, cfg=cfg, journal=Journal(), pipeline_logger=logger)

# The callback is an explicit tripwire; execute_immediately=False must prevent calls.
def forbidden_execution(*args, **kwargs):
    raise RuntimeError("execution callback invoked during code-only generation")

nodes = []
agent.update_data_preview()
for i in range(1):
    # Use normal AgentSearch steps, but disable CPU preflight/review because the
    # host MLEvolve checkout lacks its model_preflight dependency and its
    # llm.generate Codex dispatcher. Static precision/dependency checks still run;
    # execution is explicitly deferred.
    node = agent.step(exec_callback=forbidden_execution, node=None, execute_immediately=False)
    if node is None:
        break
    node.pending_execution = True
    nodes.append(node)
    prompt = node.prompt_input or ""
    source_index = counter["n"] - 1
    (prompts_dir / f"candidate_{source_index:02d}.node-prompt.md").write_text(prompt, encoding="utf-8")

summary = {
    "schema": "mlevolve-petfinder-codeonly-generation-v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": cfg.exp_name, "model": cfg.agent.code.model, "provider": cfg.agent.code.provider,
    "codex_cli_login_status_checked": True, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    "execution_callback_calls": 0, "training_launched": False, "cloud_or_provisioning": False,
    "dataset_root": str(DATA), "dataset_train_images": sum(1 for _ in (DATA / "train").glob("*.jpg")),
    "dataset_test_images": sum(1 for _ in (DATA / "test").glob("*.jpg")),
    "generated_node_count": len(list(raw_dir.glob("candidate_*.py"))),
    "review_status": "candidate_00 raw draft preserved; candidates_01+ completed MLEvolve AgentSearch code-only steps with review/preflight disabled and static checks only",
    "nodes": [{"id": n.id, "branch_id": n.branch_id, "stage": n.stage,
               "pending_execution": bool(getattr(n, "pending_execution", False)),
               "review_status": n.review_status,
               "generated_source_sha256": hashlib.sha256((raw_dir / f"candidate_{i+1:02d}.py").read_bytes()).hexdigest() if (raw_dir / f"candidate_{i+1:02d}.py").exists() else None,
               "reviewed_source_sha256": None,
               "prompt_sha256": hashlib.sha256((n.prompt_input or "").encode()).hexdigest(),
               "preflight_status": getattr(n, "preflight_status", None),
               "preflight_admitted": getattr(n, "preflight_admitted", None)} for i, n in enumerate(nodes)],
    "mlevolve_log_dir": str(run_log), "request_log": str(request_log),
}
(OUT / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
