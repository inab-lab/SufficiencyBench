#!/usr/bin/env python3
"""
Robust, resumable pipeline for completing all remaining experiments.

Design principles:
- Each step runs as a SUBPROCESS so memory is fully freed between steps
- State tracked in a JSON file — can resume from any point after crash
- Waits for any currently running processes before proceeding
- One GPU, one model at a time
- Explicit memory checks before each heavy step

Usage:
  python scripts/robust_pipeline.py [--device cuda:0] [--reset]
"""

import json
import subprocess
import sys
import os
import time
import signal
import gc
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "results" / "pipeline_state.json"
LOG_FILE = PROJECT_ROOT / "results" / "pipeline.log"

DEVICE = "cuda:0"

# Steps in execution order. Each step has:
#   check: how to verify it's already done
#   cmd: what to run
#   timeout: max seconds (0 = no timeout)
STEPS = [
    {
        "id": "wait_qwen_baselines",
        "desc": "Wait for Qwen baselines if still running",
        "type": "wait_process",
        "process_name": "baselines.py --model qwen",
        "result_check": "results/qwen/baseline_results.json",
    },
    {
        "id": "qwen_deco_rag",
        "desc": "Run DECO-RAG evaluation for Qwen",
        "result_check": "results/qwen/deco_rag_results.json",
        "cmd": [
            sys.executable, "src/methods/deco_rag.py",
            "--model", "qwen", "--device", "{device}",
            "--n_samples", "200",
        ],
        "timeout": 7200,
    },
    {
        "id": "wait_llama_download",
        "desc": "Wait for Llama download to complete",
        "type": "wait_download",
        "model_dir": "data/models/llama-3.1-8b-instruct",
        "expected_files": ["config.json", "tokenizer.json"],
        "expected_safetensors": 4,
    },
    {
        "id": "verify_llama",
        "desc": "Verify Llama model loads correctly",
        "cmd": [
            sys.executable, "-c",
            "import torch; from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig; "
            "path = '{model_dir}'; "
            "t = AutoTokenizer.from_pretrained(path); "
            "m = AutoModelForCausalLM.from_pretrained(path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map='{device}', low_cpu_mem_usage=True); "
            "print(f'OK: {{m.config.num_hidden_layers}} layers, {{m.config.hidden_size}} dim'); "
            "del m; torch.cuda.empty_cache()",
        ],
        "timeout": 300,
    },
    {
        "id": "llama_extract_states",
        "desc": "Extract Llama hidden states (all splits)",
        "result_check": "data/experiments/hidden_states/llama/test/done.flag",
        "cmd": [
            sys.executable, "src/extraction/extract_states.py",
            "--model", "llama", "--device", "{device}",
        ],
        "timeout": 28800,  # 8 hours
    },
    {
        "id": "llama_deco",
        "desc": "Run DECO analysis for Llama",
        "result_check": "results/llama/deco_results.json",
        "cmd": [
            sys.executable, "src/methods/deco.py",
            "--model", "llama",
        ],
        "timeout": 600,
    },
    {
        "id": "llama_baselines",
        "desc": "Run baselines for Llama",
        "result_check": "results/llama/baseline_results.json",
        "cmd": [
            sys.executable, "src/methods/baselines.py",
            "--model", "llama", "--device", "{device}",
        ],
        "timeout": 14400,  # 4 hours
    },
    {
        "id": "llama_deco_rag",
        "desc": "Run DECO-RAG evaluation for Llama",
        "result_check": "results/llama/deco_rag_results.json",
        "cmd": [
            sys.executable, "src/methods/deco_rag.py",
            "--model", "llama", "--device", "{device}",
            "--n_samples", "200",
        ],
        "timeout": 14400,
    },
    {
        "id": "generate_figures",
        "desc": "Regenerate all figures and tables with 3-model data",
        "cmd": [
            sys.executable, "src/evaluation/generate_figures.py",
        ],
        "timeout": 120,
        "always_run": True,  # Always regenerate
    },
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": {}, "started": datetime.now().isoformat()}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_step_done(step, state):
    """Check if a step is already completed."""
    step_id = step["id"]

    # Check state file
    if step_id in state["completed"] and not step.get("always_run"):
        return True

    # Check result file existence
    if "result_check" in step:
        result_path = PROJECT_ROOT / step["result_check"]
        if result_path.exists() and result_path.stat().st_size > 10:
            return True

    return False


def get_available_memory_gb():
    """Get available system memory in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        return 999  # assume OK if can't read


def get_gpu_free_memory_gb(device_id=0):
    """Get free GPU memory in GB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             f"--id={device_id}"],
            capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip()) / 1024
    except Exception:
        return 999


def wait_for_process(process_name, timeout=36000):
    """Wait for a process matching the name to finish."""
    log(f"Checking for running process: {process_name}")
    start = time.time()

    while time.time() - start < timeout:
        result = subprocess.run(
            ["pgrep", "-f", process_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log(f"Process '{process_name}' is not running (or has finished)")
            return True

        pids = result.stdout.strip().split("\n")
        # Filter out our own process
        my_pid = str(os.getpid())
        pids = [p for p in pids if p and p != my_pid]
        if not pids:
            log(f"Process '{process_name}' is not running")
            return True

        elapsed = int(time.time() - start)
        log(f"Waiting for PID(s) {', '.join(pids)} ({elapsed}s elapsed)...")
        time.sleep(60)

    log(f"TIMEOUT waiting for {process_name}")
    return False


def wait_for_download(model_dir, expected_files, expected_safetensors, timeout=14400):
    """Wait for model download to complete."""
    full_dir = PROJECT_ROOT / model_dir
    log(f"Waiting for download in {full_dir}")
    start = time.time()

    while time.time() - start < timeout:
        # Check if expected files exist
        config_ok = all((full_dir / f).exists() for f in expected_files)

        # Count safetensors files
        safetensors = list(full_dir.glob("*.safetensors"))
        n_safetensors = len(safetensors)

        # Check for incomplete downloads
        incomplete = list(full_dir.rglob("*.incomplete"))

        if config_ok and n_safetensors >= expected_safetensors and len(incomplete) == 0:
            total_size = sum(f.stat().st_size for f in safetensors) / 1e9
            log(f"Download complete: {n_safetensors} safetensors, {total_size:.1f} GB total")
            return True

        # Check if download process is still running
        result = subprocess.run(
            ["pgrep", "-f", "snapshot_download"],
            capture_output=True, text=True,
        )
        download_running = result.returncode == 0

        elapsed = int(time.time() - start)
        total_size = sum(f.stat().st_size for f in full_dir.rglob("*") if f.is_file()) / 1e9
        log(f"Download progress: {n_safetensors}/{expected_safetensors} safetensors, "
            f"{total_size:.1f} GB, {len(incomplete)} incomplete, "
            f"downloader {'running' if download_running else 'NOT running'} "
            f"({elapsed}s elapsed)")

        if not download_running and n_safetensors < expected_safetensors:
            # Download process died — restart it
            log("Download process not running — restarting download...")
            restart_download(full_dir)

        time.sleep(120)

    log(f"TIMEOUT waiting for download")
    return False


def restart_download(target_dir):
    """Restart the Llama download if it died."""
    cmd = [
        sys.executable, "-c",
        f"""
from huggingface_hub import snapshot_download
import os
target = '{target_dir}'
os.makedirs(target, exist_ok=True)
print('Restarting download of unsloth/Meta-Llama-3.1-8B-Instruct...')
path = snapshot_download(
    'unsloth/Meta-Llama-3.1-8B-Instruct',
    local_dir=target,
)
print(f'Downloaded to: {{path}}')
"""
    ]
    # Run in background
    proc = subprocess.Popen(
        cmd, stdout=open(LOG_FILE, "a"), stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    log(f"Restarted download as PID {proc.pid}")


def run_step_cmd(step, device):
    """Run a step's command as a subprocess."""
    cmd = []
    model_dir = str(PROJECT_ROOT / "data/models/llama-3.1-8b-instruct")
    for arg in step["cmd"]:
        arg = arg.replace("{device}", device)
        arg = arg.replace("{model_dir}", model_dir)
        cmd.append(arg)

    timeout = step.get("timeout", 3600)
    log(f"Running: {' '.join(cmd)}")
    log(f"Timeout: {timeout}s ({timeout/3600:.1f}h)")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "4",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "MALLOC_ARENA_MAX": "2",
                "SAFETENSORS_FAST_GPU": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        )

        if result.returncode != 0:
            log(f"FAILED with exit code {result.returncode}")
            return False

        log(f"SUCCESS")
        return True

    except subprocess.TimeoutExpired:
        log(f"TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        log(f"ERROR: {e}")
        return False


def check_memory_ok(min_ram_gb=5.0, min_gpu_gb=2.0, gpu_id=0):
    """Check if there's enough memory to proceed."""
    ram = get_available_memory_gb()
    gpu = get_gpu_free_memory_gb(gpu_id)
    log(f"Memory check: RAM={ram:.1f}GB available, GPU={gpu:.1f}GB free")
    if ram < min_ram_gb:
        log(f"WARNING: Low RAM ({ram:.1f}GB < {min_ram_gb}GB)")
        # Try to free some cache
        subprocess.run(["sync"], timeout=10)
        time.sleep(5)
        ram = get_available_memory_gb()
        if ram < min_ram_gb:
            log(f"Still low RAM after sync: {ram:.1f}GB")
            return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reset", action="store_true", help="Reset state and start over")
    parser.add_argument("--from-step", type=str, help="Start from this step ID")
    args = parser.parse_args()

    device = args.device
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    log("=" * 70)
    log(f"ROBUST PIPELINE — Starting at {datetime.now()}")
    log(f"Device: {device}, GPU ID: {gpu_id}")
    log("=" * 70)

    if args.reset:
        state = {"completed": {}, "started": datetime.now().isoformat()}
        save_state(state)
    else:
        state = load_state()

    skip_until = args.from_step
    skipping = bool(skip_until)

    for step in STEPS:
        step_id = step["id"]
        desc = step["desc"]

        if skipping:
            if step_id == skip_until:
                skipping = False
            else:
                log(f"SKIP (--from-step): {step_id}")
                continue

        log("")
        log(f"{'='*60}")
        log(f"STEP: {step_id} — {desc}")
        log(f"{'='*60}")

        # Check if already done
        if is_step_done(step, state):
            log(f"ALREADY DONE — skipping")
            continue

        # Memory check before heavy steps
        step_type = step.get("type", "cmd")
        if step_type == "cmd" and step.get("timeout", 0) > 300:
            if not check_memory_ok(min_ram_gb=5.0, min_gpu_gb=2.0, gpu_id=gpu_id):
                log("Insufficient memory — waiting 60s and retrying...")
                time.sleep(60)
                gc.collect()
                if not check_memory_ok(min_ram_gb=3.0, min_gpu_gb=1.0, gpu_id=gpu_id):
                    log("FATAL: Cannot proceed with this low memory")
                    sys.exit(1)

        # Execute step
        success = False
        if step_type == "wait_process":
            success = wait_for_process(step["process_name"])
            # Verify result file exists after waiting
            if success and "result_check" in step:
                result_path = PROJECT_ROOT / step["result_check"]
                if not result_path.exists():
                    log(f"WARNING: Process finished but result file missing: {result_path}")
                    log("The process may have failed. Check logs.")
                    # Don't mark as failed — maybe the file appears shortly
                    time.sleep(10)
                    if not result_path.exists():
                        log(f"Result file still missing. Marking step as failed.")
                        success = False

        elif step_type == "wait_download":
            success = wait_for_download(
                step["model_dir"],
                step["expected_files"],
                step["expected_safetensors"],
            )

        else:
            # Regular command — retry once on failure
            success = run_step_cmd(step, device)
            if not success:
                log("First attempt failed. Waiting 30s and retrying...")
                time.sleep(30)
                gc.collect()
                success = run_step_cmd(step, device)

        if success:
            state["completed"][step_id] = {
                "time": datetime.now().isoformat(),
                "status": "ok",
            }
            save_state(state)
            log(f"COMPLETED: {step_id}")
        else:
            log(f"FAILED: {step_id}")
            log("Pipeline stopped. Fix the issue and re-run — it will resume from here.")
            sys.exit(1)

        # Brief pause between heavy steps to let memory settle
        if step_type == "cmd" and step.get("timeout", 0) > 300:
            log("Pausing 10s between steps for memory cleanup...")
            time.sleep(10)

    log("")
    log("=" * 70)
    log("PIPELINE COMPLETE!")
    log("=" * 70)

    # Print summary of results
    log("\nResults summary:")
    for model in ["mistral", "qwen", "llama"]:
        model_dir = PROJECT_ROOT / "results" / model
        for fname in ["deco_results.json", "baseline_results.json", "deco_rag_results.json"]:
            fpath = model_dir / fname
            if fpath.exists():
                log(f"  {model}/{fname}: {fpath.stat().st_size} bytes")
            else:
                log(f"  {model}/{fname}: MISSING")


if __name__ == "__main__":
    main()
