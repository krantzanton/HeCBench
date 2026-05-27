#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

PRECISION = 6
TIME_PATTERN_SUM = re.compile(r"\[SYCL\]\[sum\][a-z\s]+(\d+):\s*([\d.+eE-]+)")
TIME_PATTERN_AVG = re.compile(r"\[SYCL\]\[avg\][a-z\s]+(\d+):\s*([\d.+eE-]+)")
# Matches LIKWID summary rows and captures: event name, sum, min, max, avg.
STAT_PATTERN = re.compile(
    r"\|\s*(.+?)\s+STAT\s*\|(?:\s*[^|]+\|)?\s*([0-9.+eE-]+)\s*\|\s*([0-9.+eE-]+)\s*\|\s*([0-9.+eE-]+)\s*\|\s*([0-9.+eE-]+)\s*\|"
)

GROUPS = ("fp", "l3", "stalls")
GROUP_LABELS = {"fp": "FP", "l3": "L3", "stalls": "STALLS"}


def read_text(path):
    return Path(path).read_text(errors="replace")


def discover_logs(root):
    """Return {project_name: run_log_path}. Accepts either a file, a run.log, a
    project folder, or a result root containing many project folders.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if root.is_file():
        # Useful for pasted/collected logs, such as the uploaded fp_res/l3_res/stalls_res.
        return {root.stem: root}

    if (root / "run.log").is_file():
        return {root.name: root / "run.log"}

    logs = {}
    for log in sorted(root.rglob("run.log")):
        name = log.parent.name
        logs[name] = log
    return logs


def parse_times(content):
    sums = {kernel: float(value) for kernel, value in TIME_PATTERN_SUM.findall(content)}
    avgs = {kernel: float(value) for kernel, value in TIME_PATTERN_AVG.findall(content)}
    common = sorted(set(sums) & set(avgs), key=lambda x: int(x) if x.isdigit() else x)
    return sum(sums[k] for k in common), sum(avgs[k] for k in common)


def clean_event_name(name):
    return re.sub(r"\s+", " ", name).strip()


def parse_stats(content):
    stats = {}
    for name, total, min_v, max_v, avg in STAT_PATTERN.findall(content):
        event = clean_event_name(name)
        # Keep the avg value because the old script used the STAT Avg column.
        # If duplicated rows exist, the later derived metric table wins, which is usually nicer.
        stats[event] = avg
    return stats


def fmt(value):
    try:
        return f"{float(value):.{PRECISION}f}"
    except Exception:
        return ""


def prefixed_stats(stats, backend, group):
    prefix = f"{backend} {GROUP_LABELS[group]} avg "
    return {prefix + key: fmt(value) for key, value in stats.items()}


def should_skip(contents):
    return any("[TIMEOUT]" in c or "[SKIP]" in c for c in contents if c)


def normal_parse(args):
    ocl_logs = discover_logs(args.result_ocl)
    omp_logs = discover_logs(args.result_omp)
    names = sorted(set(ocl_logs) & set(omp_logs))
    if not names:
        print("No matching projects found between OCL and OMP inputs.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).resolve()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "ocl_total", "omp_total", "ocl_avg", "omp_avg",
                "total_delta", "avg_delta", "fastest_total", "fastest_avg",
            ],
        )
        writer.writeheader()
        for name in names:
            ocl = read_text(ocl_logs[name])
            omp = read_text(omp_logs[name])
            if should_skip([ocl, omp]):
                print(f"Skipping {name}: TIMEOUT/SKIP found")
                continue
            ocl_total, ocl_avg = parse_times(ocl)
            omp_total, omp_avg = parse_times(omp)
            fastest_total = "same" if ocl_total == omp_total else ("ocl" if ocl_total < omp_total else "omp")
            fastest_avg = "same" if ocl_avg == omp_avg else ("ocl" if ocl_avg < omp_avg else "omp")
            writer.writerow({
                "name": name,
                "ocl_total": fmt(ocl_total),
                "omp_total": fmt(omp_total),
                "ocl_avg": fmt(ocl_avg),
                "omp_avg": fmt(omp_avg),
                "total_delta": fmt(abs(ocl_total - omp_total)),
                "avg_delta": fmt(abs(ocl_avg - omp_avg)),
                "fastest_total": fastest_total,
                "fastest_avg": fastest_avg,
            })
    print(f"Wrote {out_path}")


def grouped_likwid_parse(args):
    inputs = {
        "ocl": {
            "fp": args.result_ocl_fp,
            "l3": args.result_ocl_l3,
            "stalls": args.result_ocl_stalls,
        },
        "omp": {
            "fp": args.result_omp_fp,
            "l3": args.result_omp_l3,
            "stalls": args.result_omp_stalls,
        },
    }

    missing = [f"--result-{backend}-{group}" for backend, groups in inputs.items() for group, value in groups.items() if not value]
    if missing:
        print("Missing required grouped LIKWID inputs: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    logs = {backend: {group: discover_logs(path) for group, path in groups.items()} for backend, groups in inputs.items()}

    # Use the intersection so every output row has all three groups for both backends.
    names = None
    for backend in ("ocl", "omp"):
        for group in GROUPS:
            group_names = set(logs[backend][group])
            names = group_names if names is None else names & group_names
    names = sorted(names or [])

    # For single-file uploads the stems differ (fp_res/l3_res/stalls_res). Treat them as one benchmark.
    if not names and all(len(logs[b][g]) == 1 for b in ("ocl", "omp") for g in GROUPS):
        names = ["result"]
        for backend in ("ocl", "omp"):
            for group in GROUPS:
                only_path = next(iter(logs[backend][group].values()))
                logs[backend][group] = {"result": only_path}

    if not names:
        print("No matching projects found across all six grouped LIKWID inputs.", file=sys.stderr)
        for backend in ("ocl", "omp"):
            for group in GROUPS:
                print(f"  {backend}/{group}: {sorted(logs[backend][group])[:8]}", file=sys.stderr)
        sys.exit(1)

    rows = []
    all_stat_columns = []
    seen_cols = set()

    for name in names:
        contents = {backend: {group: read_text(logs[backend][group][name]) for group in GROUPS} for backend in ("ocl", "omp")}
        flat_contents = [contents[b][g] for b in ("ocl", "omp") for g in GROUPS]
        if should_skip(flat_contents):
            print(f"Skipping {name}: TIMEOUT/SKIP found")
            continue

        # Timing is normally identical across LIKWID groups for a backend; use FP as the representative.
        ocl_total, ocl_avg = parse_times(contents["ocl"]["fp"])
        omp_total, omp_avg = parse_times(contents["omp"]["fp"])

        row = {
            "name": name,
            "OCL total time": fmt(ocl_total),
            "OCL avg time": fmt(ocl_avg),
            "OMP total time": fmt(omp_total),
            "OMP avg time": fmt(omp_avg),
            "OCL total speedup": fmt(omp_total / ocl_total if ocl_total else 0),
            "OMP total speedup": fmt(ocl_total / omp_total if omp_total else 0),
            "OCL avg speedup": fmt(omp_avg / ocl_avg if ocl_avg else 0),
            "OMP avg speedup": fmt(ocl_avg / omp_avg if omp_avg else 0),
        }

        for backend_label, backend in (("OCL", "ocl"), ("OMP", "omp")):
            for group in GROUPS:
                stats = prefixed_stats(parse_stats(contents[backend][group]), backend_label, group)
                for col in stats:
                    if col not in seen_cols:
                        seen_cols.add(col)
                        all_stat_columns.append(col)
                row.update(stats)
        rows.append(row)

    base_columns = [
        "name", "OCL total time", "OCL avg time", "OMP total time", "OMP avg time",
        "OCL total speedup", "OMP total speedup", "OCL avg speedup", "OMP avg speedup",
    ]
    out_path = Path(args.output).resolve()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_columns + all_stat_columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Parse SYCL/LIKWID timing results for OCL and OMP backends.")
    ap.add_argument("--output", default="timing_results.csv", help="CSV output path (default: timing_results.csv)")

    # Old non-LIKWID timing mode.
    ap.add_argument("--result-ocl", default=".", help="OCL result root/project folder/run.log/log file")
    ap.add_argument("--result-omp", default=".", help="OMP result root/project folder/run.log/log file")

    ap.add_argument("--likwid", action="store_true", help="Parse the new grouped LIKWID results")

    # New grouped LIKWID mode: three result roots/files per backend.
    ap.add_argument("--result-ocl-fp", help="OCL FP result root/project folder/run.log/log file")
    ap.add_argument("--result-omp-fp", help="OMP FP result root/project folder/run.log/log file")
    ap.add_argument("--result-ocl-l3", help="OCL L3 result root/project folder/run.log/log file")
    ap.add_argument("--result-omp-l3", help="OMP L3 result root/project folder/run.log/log file")
    ap.add_argument("--result-ocl-stalls", help="OCL STALLS result root/project folder/run.log/log file")
    ap.add_argument("--result-omp-stalls", help="OMP STALLS result root/project folder/run.log/log file")

    args = ap.parse_args()
    if args.likwid:
        grouped_likwid_parse(args)
    else:
        normal_parse(args)


if __name__ == "__main__":
    main()
