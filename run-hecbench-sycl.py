#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import signal
import time
from pathlib import Path
from typing import List, Optional, Tuple
import fcntl

# ---------- helpers ----------

def drain(fd):
    """Read everything available without blocking"""
    chunks = []
    while True:
        try:
            data = os.read(fd, 4096)
            if not data:
                break
            chunks.append(data)
        except BlockingIOError:
            break
    return b"".join(chunks)

def run(
    cmd: List[str], cwd: Path, timeout: int, env: Optional[dict] = None
) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        # Make pipes non-blocking
        for fd in (p.stdout.fileno(), p.stderr.fileno()):
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        p.wait(timeout=timeout)
        time.sleep(0.05)
        out = drain(p.stdout.fileno()).decode()
        err = drain(p.stderr.fileno()).decode()
        return p.returncode, out, err

    except subprocess.TimeoutExpired:
        p.kill()  # kill the main process
        out, err = p.communicate()
        return 124, out, err + "\n[TIMEOUT]\n"

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Killing subprocess...", flush=True)
        p.kill()
        p.wait()
        raise


def list_targets(make_tool: str, proj_dir: Path, timeout: int) -> List[str]:
    # Parse targets from `make -qp`. This prints make database; targets are lines ending with ':'
    code, out, err = run([make_tool, "-qp"], proj_dir, timeout)
    if code != 0:
        return []
    # Avoid pattern rules; keep simple identifiers
    targets = set()
    for line in out.splitlines():
        # target: (not starting with a dot unless the target is exactly '.PHONY')
        if ":" in line and not line.startswith("\t"):
            tgt = line.split(":", 1)[0].strip()
            if " " in tgt or "/" in tgt or "%" in tgt or tgt == "":
                continue
            targets.add(tgt)
    return sorted(targets)


def has_target(targets: List[str], name: str) -> bool:
    return name in targets


def guess_executable(proj_dir: Path) -> Optional[Path]:
    # Prefer executables in ./bin (depth 1)
    candidates: List[Path] = []
    bin_dir = proj_dir / "bin"
    if bin_dir.is_dir():
        for p in bin_dir.iterdir():
            if p.is_file() and os.access(p, os.X_OK):
                candidates.append(p)

    # Also consider executables at project root (depth 1)
    for p in proj_dir.iterdir():
        if p.is_file() and os.access(p, os.X_OK):
            # Skip obvious non-targets
            if p.name.endswith((".sh", ".py", ".pl")):
                continue
            if p.suffix in (".o", ".a", ".so", ".dylib"):
                continue
            candidates.append(p)

    # If still nothing, allow depth 2 (common: build/, out/)
    for parent in (proj_dir / "build", proj_dir / "out", proj_dir / "bin"):
        if parent.is_dir():
            for p in parent.rglob("*"):
                if p.is_file() and os.access(p, os.X_OK):
                    if p.suffix in (".sh", ".py", ".pl", ".o", ".a", ".so", ".dylib"):
                        continue
                    candidates.append(p)

    if not candidates:
        return None

    # Choose the largest file (heuristic for “main” binary)
    candidates = list({c.resolve() for c in candidates})
    candidates.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return candidates[0]


def write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(description="Compile and test all *-sycl benchmarks.")
    ap.add_argument(
        "--sycl-root",
        default=".",
        help="Root directory containing *-sycl projects (default: .)",
    )
    ap.add_argument("--make", default="make", help="Make tool (default: make)")
    ap.add_argument(
        "--make-jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel jobs for make",
    )
    ap.add_argument(
        "--timeout-build",
        type=int,
        default=900,
        help="Build timeout per project in seconds (default: 900)",
    )
    ap.add_argument(
        "--extra-runs",
        type=int,
        default=0,
        help="how many extra runs it should do for the JIT runtime to stabilize",
    )
    ap.add_argument(
        "--timeout-run",
        type=int,
        default=900,
        help="Run timeout per project in seconds (default: 60)",
    )
    ap.add_argument(
        "--skip-run", action="store_true", help="Only compile (skip run/tests)"
    )
    ap.add_argument(
        "--device-filter",
        default=None,
        help="Value for SYCL_DEVICE_FILTER if you want to set it",
    )
    ap.add_argument(
        "--pattern", default="*-sycl", help="Glob to pick projects (default: *-sycl)"
    )
    ap.add_argument(
        "--results-dir", default="sycl_test_results", help="Where to store logs/results"
    )
    ap.add_argument(
        "--cflags-plus",
        default="",
        help="Append tokens to EXTRA_CFLAGS (passed to make as EXTRA_CFLAGS+=tok ...)",
    )
    ap.add_argument(
        "--likwid",
        default="",
        help="If likwid should be ran or not, argument should be passed with group likwid should use"
    )

    args = ap.parse_args()

    sycl_root = Path(args.sycl_root).resolve()
    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    # Discover projects
    projects = sorted([p for p in sycl_root.glob(args.pattern) if p.is_dir()])
    if not projects:
        print(f"No projects found under {sycl_root} matching {args.pattern}", flush=True)
        sys.exit(1)

    summary_rows = []
    start_time = time.time()

    for proj in projects:
        proj_name = proj.name
        print(f"==> {proj_name}", flush=True)

        makefile = proj / "Makefile"
        log_dir = results_root / proj_name
        build_log = log_dir / "build.log"
        run_log = log_dir / "run.log"

        compiled_ok = False
        run_ok = False
        failure_stage = None
        note = ""

        if not makefile.exists():
            failure_stage = "makefile_missing"
            note = "No Makefile"
            write_text(build_log, "[SKIP] No Makefile found.\n")
            summary_rows.append(
                {
                    "benchmark": proj_name,
                    "compile": "SKIP",
                    "run": "SKIP",
                    "failure_stage": failure_stage,
                    "note": note,
                }
            )
            continue

        # 1) Build
        build_cmd = [
            args.make,
            f"-j{args.make_jobs}",
            "CXX=acpp",
            "USE_GPU=no",
            "VENDOR=AdaptiveCpp",
        ]
        if args.cflags_plus:
            for tok in shlex.split(args.cflags_plus):
                build_cmd.append(f"EXTRA_CFLAGS+={tok}")
        code, out, err = run(build_cmd, proj, args.timeout_build)
        write_text(
            build_log,
            f"$ {' '.join(shlex.quote(c) for c in build_cmd)}\n\n[stdout]\n{out}\n\n[stderr]\n{err}\n\n[exit] {code}\n",
        )
        compiled_ok = code == 0
        if not compiled_ok:
            failure_stage = "compile"
            note = f"make exit {code}"
            # record and skip running
            write_text(run_log, "[SKIP] Build failed; not running.\n")
            summary_rows.append(
                {
                    "benchmark": proj_name,
                    "compile": "FAIL",
                    "run": "SKIP",
                    "failure_stage": failure_stage,
                    "note": note,
                }
            )
            continue

        if args.skip_run:
            summary_rows.append(
                {
                    "benchmark": proj_name,
                    "compile": "PASS",
                    "run": "SKIP",
                    "failure_stage": None,
                    "note": "run skipped",
                }
            )
            continue
        # Always run via `make run`
        ran_via = "make run"
        run_env = os.environ.copy()
        if args.device_filter:
            run_env["SYCL_DEVICE_FILTER"] = args.device_filter

        run_cmd = [args.make]
        if args.cflags_plus:
            for tok in shlex.split(args.cflags_plus):
                run_cmd.append(f"EXTRA_CFLAGS+={tok}")
        run_cmd.append("run")
        if args.likwid:
            if args.likwid == "l3":
                run_cmd.insert(0, "THESIS_L3")
            elif args.likwid == "stalls":
                run_cmd.insert(0, "THESIS_STALLS")
            else:
                run_cmd.insert(0, "THESIS_FP")
            run_cmd.insert(0, "-g")
            run_cmd.insert(0, "likwid-perfctr")

        num_runs = 1 + args.extra_runs
        all_run_logs = []
        run_ok = True
        last_run_code = 0
        last_r_out = ""
        last_r_err = ""

        for run_idx in range(num_runs):
            run_code, r_out, r_err = run(run_cmd, proj, args.timeout_run, env=run_env)
            last_run_code = run_code

            phase = "warmup" if run_idx < num_runs - 1 else "measured"

            all_run_logs.append(
                f"[run {run_idx + 1}/{num_runs}] phase={phase} via={ran_via}\n\n"
                f"[stdout]\n{r_out}\n\n"
                f"[stderr]\n{r_err}\n\n"
                f"[exit] {run_code}\n"
            )

            if run_code != 0:
                run_ok = False
                break

            last_r_out = r_out
            last_r_err = r_err

        # Full history of all runs
        write_text(
            log_dir / "run_all.log",
            "\n" + ("\n" + ("=" * 80) + "\n").join(all_run_logs)
        )

        # Only final measured run, for downstream parsing
        if run_ok:
            write_text(
                run_log,
                f"[via] {ran_via}\n\n[stdout]\n{last_r_out}\n\n[stderr]\n{last_r_err}\n\n[exit] {last_run_code}\n"
            )
        else:
            # If a run failed, it is still useful to keep failure info in run.log
            write_text(
                run_log,
                all_run_logs[-1]
            )

        if not run_ok:
            failure_stage = "run"
            note = f"{ran_via} failed on run {run_idx + 1}/{num_runs} with exit {last_run_code}"
        else:
            note = f"{ran_via} passed {num_runs} run(s)"

        summary_rows.append(
            {
                "benchmark": proj_name,
                "compile": "PASS" if compiled_ok else "FAIL",
                "run": "PASS" if run_ok else "FAIL",
                "failure_stage": failure_stage,
                "note": note,
            }
        )

    # Write summary
    csv_path = results_root / "results.csv"
    json_path = results_root / "results.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["benchmark", "compile", "run", "failure_stage", "note"]
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    json_path.write_text(json.dumps(summary_rows, indent=2))

    total = len(summary_rows)
    passed_both = sum(
        1 for r in summary_rows if r["compile"] == "PASS" and r["run"] == "PASS"
    )
    failed_compile = sum(1 for r in summary_rows if r["compile"] == "FAIL")
    failed_run = sum(
        1 for r in summary_rows if r["compile"] == "PASS" and r["run"] == "FAIL"
    )
    skipped_run = sum(1 for r in summary_rows if r["run"] == "SKIP")

    elapsed = time.time() - start_time
    print(f"\nSummary for {total} benchmarks:", flush=True)
    print(f"  PASS both:  {passed_both}", flush=True)
    print(f"  FAIL compile: {failed_compile}", flush=True)
    print(f"  FAIL run:     {failed_run}", flush=True)
    print(f"  SKIP run:     {skipped_run}", flush=True)
    print(f"Logs & results in: {csv_path.parent}", flush=True)
    print(f"Elapsed: {int(elapsed)}s", flush=True)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
