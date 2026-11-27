"""Offline tests: no ssh, no wandb, no agent. Run: python3 -m pytest autoloop/tests -q"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from autoloop.harness import command, curves, driver, evaluate, launcher, ledger, watcher
from autoloop.harness.config import load_config
from autoloop.harness.remote import dry_runner, free_slots
from autoloop.harness.spec import Gate, load_spec, validate_spec

REPO_EXPS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture
def cfg(tmp_path):
    exps = tmp_path / "autoloop" / "experiments"
    shutil.copytree(REPO_EXPS, exps)
    return load_config(overrides={
        "paths": {"agent_root": str(tmp_path / "agent_root"), "experiments": str(exps),
                  "secrets_file": str(tmp_path / "no_secrets.env")},
        "cluster": {"probe_interval_s": 0},
        "notify": {"wandb_alert": False},
    })


def _spec(cfg, name="e000-baseline-pgopt-proxy"):
    return load_spec(cfg.experiments_dir / name / "spec.yaml")


# --- spec validation ----------------------------------------------------------------------------

def test_baseline_specs_valid(cfg):
    for e in ("e000-baseline-pgopt-proxy", "e001-baseline-pgopt-full"):
        assert validate_spec(_spec(cfg, e), cfg) == []


def test_imdb_training_rejected(cfg):
    s = _spec(cfg)
    s.train_workloads = ["runs/parsed_plans3/imdb_scaled1/workload_200k_s1.json"]
    assert any("IMDB" in e for e in validate_spec(s, cfg))


def test_workload_names_never_expand_to_imdb(cfg):
    files = _spec(cfg).expanded_train_workloads(cfg)
    assert len(files) == 19 and not any("imdb" in f for f in files)


def test_tier_caps_and_reserved_args(cfg):
    s = _spec(cfg)
    s.model_type = s.model_type.replace("epochs40", "epochs120")
    s.extra_args = ["--model_dir", "/tmp/x"]
    errs = validate_spec(s, cfg)
    assert any("cap" in e for e in errs) and any("--model_dir" in e for e in errs)


def test_full_needs_parent_and_outside_paths_rejected(cfg):
    s = _spec(cfg, "e001-baseline-pgopt-full")
    s.parent = None
    s.statistics_file = "/home/someone/stats.json"
    errs = validate_spec(s, cfg, check_files=False)
    assert any("parent" in e for e in errs) and any("outside" in e for e in errs)


# --- command ------------------------------------------------------------------------------------

def test_command_writes_only_to_agent_root(cfg):
    s = _spec(cfg)
    cmd = command.train_command(cfg, s, 9)
    for flag in ("--model_dir", "--target_dir"):
        assert cmd[cmd.index(flag) + 1].startswith(str(cfg.agent_root))
    train = cmd[cmd.index("--workload_runs") + 1: cmd.index("--test_workload_runs")]
    test = cmd[cmd.index("--test_workload_runs") + 1: cmd.index("--qo_eval_after_each_epoch")]
    assert not any("imdb" in t for t in train)
    assert all("imdb_scaled1" in t for t in test) and len(test) == 8
    assert cmd[cmd.index("--wandb_group") + 1] == "zs-autoloop"


def test_scripts_use_run_exp_shared(cfg):
    out = command.outer_script(cfg, _spec(cfg), 9, gpu=1)
    assert "run_exp -m" in out and " -ex " not in out and "CUDA_VISIBLE_DEVICES=1" in out
    inner = command.inner_script(cfg, _spec(cfg), 9, cfg.agent_root / "venvs" / "x")
    assert "flock" in inner and "PYTHONPATH" in inner


# --- evaluation ---------------------------------------------------------------------------------

def _pred_csv(path: Path, rows):
    path.write_text("label,prediction,sql,hint\n" + "".join(f"{l},{p},{s},h\n" for s, l, p in rows))


def test_selection_metrics(tmp_path):
    p = tmp_path / "join_order_9v5_test_pred.csv"
    # q1: picks 2.0 (optimal 1.0); q2: picks optimal 3.0; q3 single candidate -> ignored
    _pred_csv(p, [("q1", 1.0, 0.9), ("q1", 2.0, 0.5), ("q2", 3.0, 0.1), ("q2", 5.0, 0.2), ("q3", 9.0, 1.0)])
    m = evaluate.selection_metrics(p)
    assert m["num_queries"] == 2
    assert m["total_selected_runtime"] == 5.0 and m["total_optimal_runtime"] == 4.0
    assert m["pick_rate"] == 0.5


def test_find_pred_csv_does_not_mix_benchmarks(tmp_path):
    for n in ("access_path_selection_9v5_test_pred.csv", "access_path_selection_with_aggregate_9v5_test_pred.csv"):
        (tmp_path / n).write_text("label,prediction,sql\n")
    assert evaluate.find_pred_csv(tmp_path, "access_path_selection").name.startswith("access_path_selection_9")


# --- gates --------------------------------------------------------------------------------------

def test_gate_against_real_baseline(cfg):
    base = curves.load_baseline(cfg.baselines_dir, "zs-pg_opt_pp3")
    assert curves.check_gates(base, [Gate(10, "job_ratio_mean", "<=", ref="zs-pg_opt_pp3", factor=1.0)],
                              cfg.baselines_dir)[0]["passed"] is True
    worse = [dict(r) for r in base]
    for r in worse:
        for k in list(r):
            if k.endswith("total_selected_runtime"):
                r[k] *= 2
    v = curves.check_gates(worse, [Gate(10, "job_ratio_mean", "<=", ref="zs-pg_opt_pp3", factor=1.25)],
                           cfg.baselines_dir)[0]
    assert v["passed"] is False


def test_gate_not_evaluated_before_epoch(cfg):
    rows = [{"epoch": 1, "mean_train_loss": 1.0}]
    assert curves.check_gates(rows, [Gate(5, "train_loss", "<=", value=0.1)], cfg.baselines_dir) == []


# --- slots / launcher ---------------------------------------------------------------------------

def test_free_slots_respects_busy(cfg):
    run = dry_runner(free={"c01": [0, 1], "c03": [1]})
    assert free_slots(cfg, run, busy={("c01", 0)}) == [("c01", 1), ("c03", 1)]


def test_submit_dry_run_respects_cap(cfg):
    cfg.raw["cluster"]["max_concurrent"] = 1
    log = []
    run = dry_runner(log=log, free={"c01": [0, 1], "c02": [0, 1]})
    specs = [_spec(cfg), _spec(cfg, "e001-baseline-pgopt-full")]
    started = launcher.submit(cfg, specs, run, dry_run=True)
    assert started == ["e000-baseline-pgopt-proxy-s9"]
    assert any("screen -dmS zsal-e000" in l for l in log)
    assert launcher.submit(cfg, specs, run, dry_run=True) == []  # cap reached


# --- watcher ------------------------------------------------------------------------------------

def test_watcher_kills_on_missed_gate_and_evaluates_finished(cfg, tmp_path):
    log = []
    run = dry_runner(log=log, free={"c01": [0, 1]})
    s0, s1 = _spec(cfg), _spec(cfg, "e001-baseline-pgopt-full")
    launcher.submit(cfg, [s0, s1], run, dry_run=True)
    # e000: wandb shows a diverging loss beyond the gate -> killed
    def history(c, name):
        if "e000" in name:
            return [{"epoch": e, "mean_train_loss": 1e7} for e in range(5)]
        return []
    # e001: finished with prediction files
    rd = command.run_dir(cfg, "e001-baseline-pgopt-full-s9")
    (rd / "exit_code").write_text("0")
    ev_dir = cfg.agent_root / "evaluation" / "e001-baseline-pgopt-full-s9"
    ev_dir.mkdir(parents=True)
    _pred_csv(ev_dir / "join_order_9v5_test_pred.csv", [("q", 1.0, 0.2), ("q", 2.0, 0.1)])
    events = watcher.tick(cfg, run, history)
    kinds = {e["run"]: e["type"] for e in events}
    assert kinds == {"e000-baseline-pgopt-proxy-s9": "killed", "e001-baseline-pgopt-full-s9": "finished"}
    assert launcher.read_state(cfg, "e000-baseline-pgopt-proxy-s9")["status"] == "killed"
    assert any("pkill" in l for l in log)
    assert json.loads((rd / "eval.json").read_text())["benchmarks"]["join_order"]["ratio"] == 2.0


# --- ledger -------------------------------------------------------------------------------------

def test_ledger_parses_codex_jsonl_and_enforces_budget(cfg):
    out = "\n".join([
        json.dumps({"type": "thread.started"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1_000_000, "cached_input_tokens": 400_000,
                                                         "output_tokens": 100_000}}),
    ])
    u = ledger.usage_from_output(out)
    assert u == {"input": 1_000_000, "cached_input": 400_000, "output": 100_000}
    m = cfg["agent"]["model"]
    assert ledger.price_known(cfg, m)              # conservative defaults in the repo config
    cfg.raw["budget"]["prices"][m] = {"input": 2.0, "cached_input": 0.5, "output": 10.0}
    assert ledger.cost_usd(cfg, m, u) == pytest.approx(0.6 * 2 + 0.4 * 0.5 + 0.1 * 10)
    cfg.raw["budget"].update(total_usd=5, max_usd_per_phase={"plan": 3, "analyse": 1})
    ledger.record(cfg, "plan", m, u, 1.0)
    assert ledger.spent(cfg) == pytest.approx(2.4)
    assert not ledger.can_afford_phase(cfg)


# --- driver -------------------------------------------------------------------------------------

def test_driver_dry_run_plans_round_and_submits(cfg):
    run = dry_runner(free={n: [0, 1] for n in cfg["cluster"]["nodes"]})
    r = driver.step(cfg, run, dry_run=True)
    st = driver.load_state(cfg)
    assert r == "waiting"
    assert st["rounds"]["1"]["experiments"] == ["e000-baseline-pgopt-proxy", "e001-baseline-pgopt-full"]
    assert set(launcher.active_runs(cfg)) == {"e000-baseline-pgopt-proxy-s9", "e001-baseline-pgopt-full-s9"}
    prompts = list((cfg.state_dir / "agent_logs").glob("*plan-r1.prompt.md"))
    assert prompts and "{{" not in prompts[0].read_text()


def test_pause_and_review_gate(cfg):
    cfg.state_dir.mkdir(parents=True)
    (cfg.state_dir / "STOP").touch()
    assert driver.step(cfg, dry_runner(), dry_run=True) == "paused"
    st = driver.load_state(cfg)
    st.update(phase="AWAITING_REVIEW", review_started=9e12)
    driver.save_state(cfg, st)
    assert "REVIEW.md" in driver.resume(cfg)
    assert driver.resume(cfg, force=True) == "resumed"
    assert not (cfg.state_dir / "STOP").exists()


# --- subplex wiring and per-benchmark scoring ---------------------------------------------------

def test_subplex_analysis_command(cfg):
    from autoloop.harness import subplex_analysis
    s = _spec(cfg)
    cmd = subplex_analysis.analysis_command(cfg, s, 9)
    assert cmd[cmd.index("-m") + 1] == "subplex.external_model"
    assert cmd[cmd.index("--architecture") + 1] == s.model_type
    assert cmd[cmd.index("--reference") + 1] == cfg["subplex"]["reference_pp3"]
    assert cmd[cmd.index("--out-root") + 1].startswith(str(cfg.agent_root))
    assert cmd[cmd.index("--code-src") + 1].endswith("e000-baseline-pgopt-proxy-s9/code/src")
    assert "--batch-independent" not in cmd and "--ppv50" not in cmd   # RMP model on parsed_plans3
    s.plans_dir = "runs/parsed_plans50"
    cmd = subplex_analysis.analysis_command(cfg, s, 9)
    assert "--ppv50" in cmd and cmd[cmd.index("--reference") + 1] == cfg["subplex"]["reference_ppv50"]


def test_per_benchmark_verdicts_and_geo_mean(cfg, tmp_path):
    ev = tmp_path / "ev"
    ev.mkdir()
    _pred_csv(ev / "join_order_9v5_test_pred.csv", [("q", 1.0, 0.2), ("q", 2.0, 0.1)])            # ratio 2.0
    _pred_csv(ev / "physical_operator_9v5_test_pred.csv", [("q", 1.0, 0.1), ("q", 2.0, 0.2)])     # ratio 1.0
    ref = {"ratio": {"join_order": 1.5, "physical_operator": 1.2}, "imdb_200k_median_qerror": 1.25}
    r = evaluate.evaluate_dir(cfg, ev, ref)
    assert r["vs_reference"]["join_order"]["verdict"] == "worse"
    assert r["vs_reference"]["physical_operator"]["verdict"] == "better"
    assert (r["wins"], r["losses"]) == (1, 1)
    assert r["score"] == pytest.approx(2 ** 0.5)          # geometric mean of 2.0 and 1.0


def test_leaderboard_shows_every_benchmark(cfg, tmp_path):
    exp = cfg.experiments_dir / "e000-baseline-pgopt-proxy"
    run = {"score": 1.2, "ratios": {"join_order": 1.0, "job_w_optimal_w_idxs": 3.0},
           "imdb_200k_median_qerror": 1.3}
    evaluate.write_result(cfg, exp, {"e000-baseline-pgopt-proxy-s9": run}, evaluate.load_reference(cfg))
    board = evaluate.leaderboard(cfg)
    assert "| reference |" in board and "JOB-light" in board and "AP-agg" in board
    assert "1.000 ↓" in board and "3.000 ↑" in board and "1/1" in board


def test_inner_script_pins_shared_interpreter(cfg):
    inner = command.inner_script(cfg, _spec(cfg), 9, cfg.agent_root / "venvs" / "x")
    assert f"--python {cfg['cluster']['python']}" in inner and "UV_PYTHON_INSTALL_DIR" in inner


# --- framework enforcement ----------------------------------------------------------------------

def _agent_exp(cfg, hypothesis_text):
    d = cfg.experiments_dir / "e002-log-card"
    d.mkdir()
    spec = yaml.safe_load((cfg.experiments_dir / "e000-baseline-pgopt-proxy" / "spec.yaml").read_text())
    spec.update(id="e002-log-card", branch="agent/e002-log-card")
    (d / "spec.yaml").write_text(yaml.safe_dump(spec))
    (d / "hypothesis.md").write_text(hypothesis_text)
    return load_spec(d / "spec.yaml")


def test_hypothesis_without_framework_sections_is_rejected(cfg):
    s = _agent_exp(cfg, "# e002\nLog cards should help.\n")
    errs = validate_spec(s, cfg)
    assert any("## Step 1" in e for e in errs) and any("## Predicted framework change" in e for e in errs)


def test_template_shaped_hypothesis_is_accepted(cfg):
    body = "x" * 60
    text = "\n".join(f"{h}\n{body}\n" for h in
                     ["## Step 1 - where", "## Step 2 - what", "## Step 3 - why", "## Change",
                      "## Predicted framework change"])
    assert validate_spec(_agent_exp(cfg, text), cfg) == []


def test_leaderboard_marks_unexplained(cfg):
    exp = cfg.experiments_dir / "e000-baseline-pgopt-proxy"
    evaluate.write_result(cfg, exp, {"e000-baseline-pgopt-proxy-s9": {"score": 1.2, "ratios": {"join_order": 1.0}}},
                          evaluate.load_reference(cfg))
    assert "NO - not a result yet" in evaluate.leaderboard(cfg)
    (exp / "analysis.md").write_text("\n".join(f"{h}\n{'y' * 60}\n" for h in
                                               ["## Verdict", "## Step 1", "## Step 2", "## Step 3", "## Explanation", "## Framework value"]))
    assert "| yes |" in evaluate.leaderboard(cfg)


def test_inner_script_reuses_prebuilt_venv(cfg):
    inner = command.inner_script(cfg, _spec(cfg), 9, cfg.agent_root / "venvs" / "x")
    assert "[ -x " in inner and "|| flock" in inner


# --- agent sandbox (Landlock) -------------------------------------------------------------------

def test_sandbox_blocks_secrets_and_allows_repo(tmp_path):
    import subprocess, sys
    from autoloop.harness import sandbox
    if sandbox.abi_version() < 4:
        pytest.skip("landlock ABI 4 not available")
    secret = Path.home() / ".config" / "zs-autoloop" / "secrets.env"
    allowed = tmp_path / "ok.txt"
    code = ("import sys\n"
            f"open({str(allowed)!r}, 'w').write('x')\n"
            f"try:\n    open({str(secret)!r}).read(); sys.exit(3)\n"
            "except PermissionError:\n    sys.exit(0)\n"
            "except FileNotFoundError:\n    sys.exit(0)\n")
    r = subprocess.run([sys.executable, "-m", "autoloop.harness.sandbox", "--ro", "/usr", "--ro", "/lib",
                        "--ro", "/lib64", "--ro", "/etc", "--ro", "/proc", "--ro", str(Path.home() / ".local/share/uv"),
                        "--ro", sys.prefix, "--ro", sys.base_prefix,
                        "--ro", str(Path(__file__).resolve().parents[2]), "--rw", str(tmp_path), "--rw", "/dev",
                        "--", sys.executable, "-c", code],
                       cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert allowed.read_text() == "x"


# --- framework usage tracing --------------------------------------------------------------------

def test_framework_trace_classifies_codex_commands(cfg):
    from autoloop.harness import framework_trace as ft
    log = "\n".join(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": c,
                                                                    "exit_code": 0}}) for c in [
        "cat /mnt/labstore/vweidinger/agent_zs-autoloop/subplex/e002-s9/report/summary.json",
        "head subplex/_reference/ref-zs-pg_opt/report/decisive_job_w_optimal_w_idxs.csv",
        "uv run python -m subplex.step4.emm --model e002-s9 --dataset subplex_join_order",
        "sed -n 1,80p ~/subplex/tmp_with_jit/step6/REPORT.md",
        "uv run python - <<EOF\nfrom subplex.step7.training_evidence import evidence_for\nEOF",
        "git status --short",
    ])
    steps = [r["step"] for r in ft.extract(log)]
    assert steps == ["step1-where", "step1-where", "step2-what (step4)", "step3-why (step5/6)",
                     "step3-training (step7)"]
    ft.record_phase(cfg, "analyse-e002-x", "20261126-000000", log)
    rep = ft.report(cfg)
    assert "analyse-e002-x | 2 | 1 | 1 | 1 |" in rep and "e000-baseline-pgopt-proxy" in rep


# --- circuit breakers: no money is spent on tooling problems ------------------------------------

def _finished_round(cfg, monkeypatch):
    """Round 1 planned with e000, its run finished; the agent and subplex are stubbed."""
    st = driver.load_state(cfg)
    st["rounds"]["1"] = {"planned": True, "experiments": ["e000-baseline-pgopt-proxy"], "analysed": []}
    driver.save_state(cfg, st)
    launcher.write_state(cfg, "e000-baseline-pgopt-proxy-s9", status="finished", experiment="e000-baseline-pgopt-proxy")
    (cfg.agent_root / "models" / "e000-baseline-pgopt-proxy-s9").mkdir(parents=True)
    (cfg.agent_root / "models" / "e000-baseline-pgopt-proxy-s9" / "m_9.pt").write_text("x")
    monkeypatch.setattr(watcher, "tick", lambda *a, **k: [])
    monkeypatch.setattr(driver, "_local_gpu", lambda *a: "")
    monkeypatch.setattr(driver, "push_progress", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_commit_generated", lambda *a, **k: None)


def test_halt_when_subplex_fails(cfg, monkeypatch):
    from autoloop.harness import agent, subplex_analysis
    _finished_round(cfg, monkeypatch)
    calls = []
    monkeypatch.setattr(subplex_analysis, "run_analysis", lambda *a, **k: {"returncode": 1, "log": "x.log"})
    monkeypatch.setattr(agent, "run_phase", lambda *a, **k: calls.append(a) or {"returncode": 0})
    assert driver.step(cfg, dry_runner()) == "awaiting_review"
    assert calls == [] and driver.load_state(cfg)["stop_reason"] == "subplex analysis failed"


def test_halt_on_agent_error(cfg, monkeypatch):
    from autoloop.harness import agent, subplex_analysis
    _finished_round(cfg, monkeypatch)
    monkeypatch.setattr(subplex_analysis, "run_analysis", lambda *a, **k: {"returncode": 0})
    monkeypatch.setattr(agent, "run_phase", lambda *a, **k: {"returncode": 1, "log": "y"})
    assert driver.step(cfg, dry_runner()) == "awaiting_review"
    assert driver.load_state(cfg)["stop_reason"] == "agent error"


def test_budget_stop_is_clean_and_resumable(cfg, monkeypatch):
    from autoloop.harness import agent, subplex_analysis
    _finished_round(cfg, monkeypatch)
    monkeypatch.setattr(subplex_analysis, "run_analysis", lambda *a, **k: {"returncode": 0})
    calls = []
    monkeypatch.setattr(agent, "run_phase", lambda *a, **k: calls.append(a) or {"returncode": 0})
    cfg.raw["budget"].update(total_usd=5, max_usd_per_phase=10)
    assert driver.step(cfg, dry_runner()) == "awaiting_review"
    assert calls == [] and driver.load_state(cfg)["stop_reason"] == "budget"
    cfg.raw["budget"]["total_usd"] = 100                       # raise the limit ...
    assert driver.resume(cfg, force=True) == "resumed"         # ... and continue: nothing is repeated
    driver.step(cfg, dry_runner())
    assert len(calls) == 1 and "stop_reason" not in driver.load_state(cfg)


def test_agent_sandbox_profile_keeps_harness_read_only(cfg):
    from autoloop.harness import sandbox
    prof = sandbox.agent_profile(cfg)
    assert str(cfg.repo) in prof["ro"] and str(cfg.repo) not in prof["rw"]
    assert str(cfg.experiments_dir) in prof["rw"] and str(cfg.repo / "worktrees") in prof["rw"]
    assert not any("harness" in p or "config" in p for p in prof["rw"])



def test_session_usage_reads_live_codex_totals(tmp_path):
    from autoloop.harness import agent
    d = tmp_path / "2026" / "11" / "26"
    d.mkdir(parents=True)
    ev = lambda i, c, o: json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": i, "cached_input_tokens": c, "output_tokens": o}}}})
    (d / "rollout-x.jsonl").write_text("\n".join([ev(100, 50, 5), '{"type":"other"}', ev(1000, 900, 30)]) + "\n")
    assert agent.session_usage(0, tmp_path) == {"input": 1000, "cached_input": 900, "output": 30}
    assert agent.session_usage(9e12, tmp_path) is None


def test_phase_limits_by_kind(cfg):
    cfg.raw["budget"]["max_usd_per_phase"] = {"plan": 20, "analyse": 8}
    assert ledger.phase_limit(cfg, "plan-r1") == 20 and ledger.phase_limit(cfg, "analyse-e002-x") == 8


# --- fixes after the first real round -------------------------------------------------------------

def test_run_scripts_need_no_uv_and_report_the_real_exit_code(cfg):
    inner = command.inner_script(cfg, _spec(cfg), 9, cfg.agent_root / "venvs" / "x")
    run_line = [l for l in inner.splitlines() if "main.py" in l][0]
    assert run_line.startswith(str(cfg.agent_root / "venvs" / "x" / "bin" / "python")) and "uv run" not in inner
    assert "inner_exit_code" in inner
    outer = command.outer_script(cfg, _spec(cfg), 9, gpu=0)
    assert "inner_exit_code ]; then cat" in outer


def test_exit_zero_without_predictions_is_a_failure(cfg):
    run = dry_runner(free={"c01": [0]})
    launcher.submit(cfg, [_spec(cfg)], run, dry_run=True)
    rd = command.run_dir(cfg, "e000-baseline-pgopt-proxy-s9")
    (rd / "exit_code").write_text("0")                   # e.g. 'uv: command not found' behind run_exp
    ev = watcher.tick(cfg, run, lambda c, n: [])
    assert ev[0]["type"] == "failed" and "no test predictions" in ev[0]["reason"]


def test_replan_adds_to_the_round_and_halts_when_empty(cfg, monkeypatch):
    from autoloop.harness import agent
    st = driver.load_state(cfg)
    st["rounds"]["1"] = {"planned": False, "experiments": ["e000-baseline-pgopt-proxy", "e001-baseline-pgopt-full"],
                         "analysed": []}
    driver.save_state(cfg, st)
    monkeypatch.setattr(watcher, "tick", lambda *a, **k: [])
    monkeypatch.setattr(driver, "push_progress", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_commit_generated", lambda *a, **k: None)
    monkeypatch.setattr(agent, "run_phase", lambda *a, **k: {"returncode": 0})
    assert driver.step(cfg, dry_runner()) == "awaiting_review"          # the agent planned nothing new
    st = driver.load_state(cfg)
    assert st["stop_reason"] == "plan without experiments"
    assert st["rounds"]["1"]["experiments"] == ["e000-baseline-pgopt-proxy", "e001-baseline-pgopt-full"]
