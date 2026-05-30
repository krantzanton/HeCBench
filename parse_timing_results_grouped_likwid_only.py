#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

PRECISION = 6

# Matches LIKWID summary rows and captures: event name, sum, min, max, avg.
STAT_PATTERN = re.compile(
    r"\|\s*(.+?)\s+STAT\s*\|(?:\s*[^|]+\|)?\s*([0-9.+eE-]+)\s*\|\s*([0-9.+eE-]+)\s*\|\s*([0-9.+eE-]+)\s*\|\s*([0-9.+eE-]+)\s*\|"
)

GROUPS = ("fp", "l3", "stalls")
GROUP_LABELS = {"fp": "fp", "l3": "l3", "stalls": "stalls"}
BACKENDS = ("ocl", "omp")


def read_text(path: Path) -> str:
    return Path(path).read_text(errors="replace")


def discover_logs(root):
    """Return {project_name: run_log_path}.

    Accepts:
      - a single log file
      - a project directory containing run.log
      - a result root containing many project directories with run.log files
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if root.is_file():
        return {root.stem: root}

    if (root / "run.log").is_file():
        return {root.name: root / "run.log"}

    logs = {}
    for log in sorted(root.rglob("run.log")):
        logs[log.parent.name] = log
    return logs


def clean_event_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def clean_column_part(name: str) -> str:
    """Make LIKWID event names CSV/header friendly but still readable."""
    name = clean_event_name(name).lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def parse_likwid_stats(content: str) -> dict[str, str]:
    stats = {}
    for name, _total, _min_v, _max_v, avg in STAT_PATTERN.findall(content):
        event = clean_event_name(name)
        stats[event] = fmt(avg)
    return stats


def fmt(value) -> str:
    try:
        return f"{float(value):.{PRECISION}f}"
    except Exception:
        return ""


def should_skip(contents) -> bool:
    return any("[TIMEOUT]" in c or "[SKIP]" in c for c in contents if c)


def collect_likwid_rows(args):
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

    missing = [
        f"--result-{backend}-{group}"
        for backend, groups in inputs.items()
        for group, value in groups.items()
        if not value
    ]
    if missing:
        print("Missing required LIKWID inputs: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    logs = {
        backend: {group: discover_logs(path) for group, path in groups.items()}
        for backend, groups in inputs.items()
    }

    names = None
    for backend in BACKENDS:
        for group in GROUPS:
            group_names = set(logs[backend][group])
            names = group_names if names is None else names & group_names
    names = sorted(names or [])

    # Single-file convenience fallback: if all six inputs are single logs with
    # different stems, treat them as one synthetic result row.
    if not names and all(len(logs[b][g]) == 1 for b in BACKENDS for g in GROUPS):
        names = ["result"]
        for backend in BACKENDS:
            for group in GROUPS:
                only_path = next(iter(logs[backend][group].values()))
                logs[backend][group] = {"result": only_path}

    if not names:
        print("No matching projects found across all six LIKWID inputs.", file=sys.stderr)
        for backend in BACKENDS:
            for group in GROUPS:
                print(f"  {backend}/{group}: {sorted(logs[backend][group])[:8]}", file=sys.stderr)
        sys.exit(1)

    likwid_by_name = {}
    stat_columns = []
    seen_columns = set()

    for name in names:
        contents = {
            backend: {
                group: read_text(logs[backend][group][name])
                for group in GROUPS
            }
            for backend in BACKENDS
        }
        flat_contents = [contents[b][g] for b in BACKENDS for g in GROUPS]
        if should_skip(flat_contents):
            print(f"Skipping LIKWID stats for {name}: TIMEOUT/SKIP found")
            continue

        row = {}
        for backend in BACKENDS:
            for group in GROUPS:
                stats = parse_likwid_stats(contents[backend][group])
                for event, value in stats.items():
                    col = f"{backend}_{GROUP_LABELS[group]}_avg_{clean_column_part(event)}"
                    if col not in seen_columns:
                        seen_columns.add(col)
                        stat_columns.append(col)
                    row[col] = value

        likwid_by_name[name] = row

    return likwid_by_name, stat_columns


def read_timing_csv(path: Path):
    if not path.exists():
        print(f"Timing CSV does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Timing CSV has no header: {path}", file=sys.stderr)
            sys.exit(1)
        if "name" not in reader.fieldnames:
            print(f"Timing CSV must contain a 'name' column: {path}", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)
    return rows, list(reader.fieldnames)


def merge_likwid_into_timing(args):
    timing_path = Path(args.timing_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    timing_rows, timing_columns = read_timing_csv(timing_path)
    likwid_by_name, stat_columns = collect_likwid_rows(args)

    # Keep all timing rows. Add LIKWID columns when the name matches; leave blank otherwise.
    merged_rows = []
    matched = set()
    for timing_row in timing_rows:
        name = timing_row.get("name", "")
        extra = likwid_by_name.get(name, {})
        if extra:
            matched.add(name)
        merged_row = dict(timing_row)
        for col in stat_columns:
            merged_row[col] = extra.get(col, "")
        merged_rows.append(merged_row)

    if args.include_likwid_only:
        for name in sorted(set(likwid_by_name) - matched):
            row = {col: "" for col in timing_columns}
            row["name"] = name
            for col in stat_columns:
                row[col] = likwid_by_name[name].get(col, "")
            merged_rows.append(row)
            matched.add(name)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timing_columns + stat_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    unmatched_timing = sorted({row.get("name", "") for row in timing_rows} - matched)
    unmatched_likwid = sorted(set(likwid_by_name) - matched)

    print(f"Read timing CSV: {timing_path}")
    print(f"Wrote merged CSV: {output_path}")
    print(f"Added LIKWID columns: {len(stat_columns)}")
    print(f"Matched projects: {len(matched)}")
    if unmatched_timing:
        print(f"Timing rows without LIKWID data: {len(unmatched_timing)}")
    if unmatched_likwid:
        print(f"LIKWID projects not present in timing CSV: {len(unmatched_likwid)}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge LIKWID FP/L3/STALLS stats into an existing timing_results.csv."
    )
    ap.add_argument(
        "--timing-csv",
        default="timing_results.csv",
        help="CSV produced by the timing parser. It must contain a 'name' column. Default: timing_results.csv",
    )
    ap.add_argument(
        "--output",
        default="timing_results_likwid.csv",
        help="Merged CSV output path. Default: timing_results_likwid.csv",
    )
    ap.add_argument(
        "--include-likwid-only",
        action="store_true",
        help="Also write rows that exist in LIKWID inputs but not in the timing CSV.",
    )

    ap.add_argument("--result-ocl-fp", required=True, help="OCL FP result root/project folder/run.log/log file")
    ap.add_argument("--result-omp-fp", required=True, help="OMP FP result root/project folder/run.log/log file")
    ap.add_argument("--result-ocl-l3", required=True, help="OCL L3 result root/project folder/run.log/log file")
    ap.add_argument("--result-omp-l3", required=True, help="OMP L3 result root/project folder/run.log/log file")
    ap.add_argument("--result-ocl-stalls", required=True, help="OCL STALLS result root/project folder/run.log/log file")
    ap.add_argument("--result-omp-stalls", required=True, help="OMP STALLS result root/project folder/run.log/log file")

    args = ap.parse_args()
    merge_likwid_into_timing(args)


if __name__ == "__main__":
    main()
