#!/usr/bin/env python3
import sys
import re
from pathlib import Path

# ------------ Config ------------
SOURCE_EXTS = {".cpp", ".cc", ".cxx", ".h", ".hpp"}
HEADER_NAME = "sycl_timer.hpp"

SYCL_TIMER_HPP = r"""#pragma once
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef SYCL_TIMER_ONLY_AGG
#define SYCL_TIMER_ONLY_AGG 1
#endif

namespace sycl_timer_detail {

inline const char* out_path() {
  if (const char* p = std::getenv("SYCL_TIMER_OUT")) return p;
  return "sycl_timing.csv";
}

inline std::mutex& mtx() { static std::mutex m; return m; }

struct Agg { double sum_ms = 0.0; long long count = 0; };

inline std::unordered_map<std::string, Agg>& agg_map() {
  static std::unordered_map<std::string, Agg> m;
  return m;
}

inline void record_agg(const std::string& label, double ms) {
  std::lock_guard<std::mutex> lock(mtx());
  auto& a = agg_map()[label];
  a.sum_ms += ms;
  a.count  += 1;
}

inline void dump_aggregates() {
  std::lock_guard<std::mutex> lock(mtx());
  if (agg_map().empty()) return;

  // CSV: write sum and avg
  if (std::FILE* f = std::fopen(out_path(), "a")) {
    for (const auto& kv : agg_map()) {
      const auto& label = kv.first;
      const auto& a = kv.second;
      const double avg = (a.count>0) ? (a.sum_ms / double(a.count)) : 0.0;
      std::fprintf(f, "%s(sum),%.6f,%lld\n", label.c_str(), a.sum_ms/1000.0, a.count);
      std::fprintf(f, "%s(avg),%.6f,%lld\n", label.c_str(), avg/1000.0, a.count);
    }
    std::fclose(f);
  }

  // Terminal: only aggregates
  for (const auto& kv : agg_map()) {
    const auto& label = kv.first;
    const auto& a = kv.second;
    const double avg = (a.count>0) ? (a.sum_ms / double(a.count)) : 0.0;
    std::fprintf(stdout, "[SYCL][sum] %s: %.6f s over %lld iters\n",
                 label.c_str(), a.sum_ms/1000.0, a.count);
    std::fprintf(stdout, "[SYCL][avg] %s: %.6f s over %lld iters\n",
                 label.c_str(), avg/1000.0, a.count);
  }
  std::fflush(stdout);
}

struct AggDumper {
  AggDumper() { std::atexit(&dump_aggregates); }
  ~AggDumper() { dump_aggregates(); }
};
static AggDumper _sycl_timer_agg_dumper_guard{};

inline void dump_now() { dump_aggregates(); }

inline double duration_ms_from_event(const sycl::event& e) {
  try {
    auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(t - s) / 1.0e6;
  } catch (...) { return -1.0; }
}

template <class F>
inline sycl::event time_expr_agg(const char* label, F&& f) {
  auto t0 = std::chrono::high_resolution_clock::now();
  sycl::event evt = std::forward<F>(f)();
  { std::vector<sycl::event> __ev{evt}; sycl::event::wait(__ev); }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = duration_ms_from_event(evt);
  if (ms < 0.0) ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  record_agg(label, ms);
  return evt;
}

inline void record_event_ms(const char* label, const sycl::event& evt) {
  { std::vector<sycl::event> __ev{evt}; sycl::event::wait(__ev); }
  double ms = duration_ms_from_event(evt);
  if (ms < 0.0) return; // profiling unavailable; skip (keeps behavior minimal)
  record_agg(label, ms);
}

} // namespace sycl_timer_detail

#define SYCL_TIME_EVT(LABEL, EVT) (::sycl_timer_detail::record_event_ms((LABEL),(EVT)))
#define SYCL_TIMER_DUMP() ::sycl_timer_detail::dump_now()


#define SYCL_TIME_AGG(LABEL, EXPR) \
  (::sycl_timer_detail::time_expr_agg((LABEL), [&](){ return (EXPR); }))

// For safety: SYCL_TIME behaves identically (aggregate-only)
#ifndef SYCL_TIME
#define SYCL_TIME(LABEL, EXPR) SYCL_TIME_AGG(LABEL, EXPR)
#endif

"""

# ------------ Patterns ------------
INCLUDE_LINE_RE = re.compile(r"^\s*#\s*include[^\n]*$", re.MULTILINE)

# q.submit / q.parallel_for / q.single_task on a queue-like object (NOT handler)
CALL_PATTERNS = [
    re.compile(
        r"(?P<obj>[A-Za-z_]\w*)\s*(?:->|\.)\s*"  # object (q or cgh)
        r"(?P<api>parallel_for|single_task|submit)"  # API
        r"\s*(?P<rest><[^;(){}]*?>\s*)?\("  # optional <...> then '('
    )
]


SYCL_TIME_CALL_RE = re.compile(r"\bSYCL_TIME\s*\(")
RETURN0_RE = re.compile(r"return\s+0\s*;")
HAS_INCLUDE_RE = re.compile(r"^\s*#\s*include\b", re.MULTILINE)
HAS_PRAGMA_RE = re.compile(r"^\s*#\s*pragma\b", re.MULTILINE)
HAS_MAIN_RE = re.compile(r"\bint\s+main\s*\(")
WAIT_TAIL_RE = re.compile(r"\.\s*wait\s*\(\s*\)\s*$")
RETURN_EXIT_RE = re.compile(r"return\s+(?:0|EXIT_SUCCESS|EXIT_FAILURE)\s*;")
RETURN_ANY_RE = re.compile(r"\breturn\b[^;]*;")


# ------------ Helpers ------------
def ensure_timer_include(src: str, path: Path) -> str:
    # never touch .sycl files
    if path.suffix.lower() == ".sycl":
        return src
    if "sycl_timer.hpp" in src:
        return src

    lines = src.splitlines(keepends=True)
    n = len(lines)

    i = 0
    in_block_comment = False

    def is_blank_or_line_comment(s: str) -> bool:
        t = s.lstrip()
        return t == "" or t.startswith("//")

    def is_include(s: str) -> bool:
        return s.lstrip().startswith("#include")

    def is_non_include_pp(s: str) -> bool:
        t = s.lstrip()
        if not t.startswith("#"):
            return False
        return not t.startswith("#include")

    # skip shebang/BOM-like
    if i < n and lines[i].startswith("#!"):
        i += 1

    # walk until we hit the first non-include preprocessor directive or real code
    insert_at = 0
    while i < n:
        s = lines[i]

        # handle block comments
        if in_block_comment:
            if "*/" in s:
                in_block_comment = False
            i += 1
            continue
        if "/*" in s and "*/" not in s:
            in_block_comment = True
            i += 1
            continue

        if is_blank_or_line_comment(s):
            i += 1
            continue

        if is_include(s):
            i += 1
            continue

        # stop BEFORE first non-include preprocessor directive (#ifdef/#if/#define/#pragma/...)
        if is_non_include_pp(s):
            break

        # stop before any real code
        break

    insert_at = sum(len(l) for l in lines[:i])
    return src[:insert_at] + '#include "../sycl_timer.hpp"\n' + src[insert_at:]


def strip_trailing_wait(s: str) -> str:
    return WAIT_TAIL_RE.sub("", s.rstrip())


def already_wrapped(src: str, idx: int) -> bool:
    start = max(0, idx - 96)
    window = src[start:idx]
    return (
        "SYCL_TIME(" in window or "SYCL_TIME_AGG(" in window or "__sycl_evt_" in window
    )


def find_statement_end(src: str, open_paren_pos: int) -> int:
    """Track (), {}, [] while skipping strings/comments; return index of the statement-ending ';'."""
    i = open_paren_pos
    n = len(src)
    paren = brace = brack = 0
    in_sq = in_dq = in_sl = in_ml = False
    while i < n:
        c = src[i]
        if in_sl:
            if c == "\n":
                in_sl = False
        elif in_ml:
            if c == "*" and i + 1 < n and src[i + 1] == "/":
                in_ml = False
                i += 1
        elif in_sq:
            if c == "\\":
                i += 1
            elif c == "'":
                in_sq = False
        elif in_dq:
            if c == "\\":
                i += 1
            elif c == '"':
                in_dq = False
        else:
            if c == "/" and i + 1 < n:
                if src[i + 1] == "/":
                    in_sl = True
                    i += 1
                elif src[i + 1] == "*":
                    in_ml = True
                    i += 1
            elif c == "'":
                in_sq = True
            elif c == '"':
                in_dq = True
            elif c == "(":
                paren += 1
            elif c == ")":
                paren -= 1
            elif c == "{":
                brace += 1
            elif c == "}":
                brace -= 1
            elif c == "[":
                brack += 1
            elif c == "]":
                brack -= 1
            elif c == ";" and paren == 0 and brace == 0 and brack == 0:
                return i
        i += 1
    return -1


# ------------ Instrumentation ------------
def instrument_source(text: str, file: Path) -> str:
    src = text

    offset = 0
    call_idx = 0  # label kernels as "kernel 1", "kernel 2", ... per file

    for pat in CALL_PATTERNS:
        while True:
            m = pat.search(src, pos=offset)
            if not m:
                break

            obj = m.group("obj") if "obj" in m.re.groupindex else None

            if obj in {"cgh", "h", "handler"}:
                offset = m.end()
                continue

            if already_wrapped(src, m.start()):
                offset = m.end()
                continue

            open_paren_pos = m.end() - 1
            if src[open_paren_pos] != "(":
                open_paren_pos = src.find("(", m.end())
                if open_paren_pos == -1:
                    offset = m.end()
                    continue

            stmt_end = find_statement_end(src, open_paren_pos)
            if stmt_end == -1:
                offset = m.end()
                continue

            # Label by callsite order (robust; no loop detection needed)
            inner = src[m.start() : stmt_end]  # without trailing ';'
            inner = strip_trailing_wait(inner)

            if HAS_INCLUDE_RE.search(inner) or HAS_PRAGMA_RE.search(inner):
                call_idx += 1
                label = f"kernel {call_idx}"
                tmp = f"__sycl_evt_k{call_idx}"
                wrapped = f'auto {tmp} = {inner}; SYCL_TIME_AGG("{label}", {tmp});'
            else:
                call_idx += 1
                label = f"kernel {call_idx}"
                wrapped = f'SYCL_TIME_AGG("{label}", {inner});'

            src = src[: m.start()] + wrapped + src[stmt_end + 1 :]
            offset = m.start() + len(wrapped)

    # Harden: convert any stray SYCL_TIME(...) -> SYCL_TIME_AGG(...)
    replacements = []
    for m in SYCL_TIME_CALL_RE.finditer(src):
        i = m.start()
        if src[i : i + 15] != "SYCL_TIME_AGG(":
            replacements.append(i)
    for i in reversed(replacements):
        src = src[:i] + "SYCL_TIME_AGG(" + src[i + 10 :]

    if src != text and file.suffix.lower() != ".sycl" and "sycl_timer.hpp" not in src:
        src = ensure_timer_include(src, file)

    src = ensure_dump_call(src)
    src = re.sub(r"auto\s+(__sycl_evt_\w+)\s*=\s*auto\s+\1\s*=", r"auto \1 = ", src)
    return src


def write_or_skip(path: Path, new_text: str) -> bool:
    old = path.read_text(encoding="utf-8", errors="ignore")
    if old == new_text:
        return False
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(old, encoding="utf-8")
    path.write_text(new_text, encoding="utf-8")
    return True


def _find_main_span(src: str):
    """Return (lb, rb) byte indices for the body of int main(...){...} or (None, None)."""
    m = re.search(r"\bint\s+main\s*\([^)]*\)\s*{", src)
    if not m:
        return (None, None)
    i = m.end() - 1  # at '{'
    n = len(src)
    depth = 0
    in_sq = in_dq = in_sl = in_ml = False
    while i < n:
        c = src[i]
        if in_sl:
            if c == "\n":
                in_sl = False
        elif in_ml:
            if c == "*" and i + 1 < n and src[i + 1] == "/":
                in_ml = False
                i += 1
        elif in_sq:
            if c == "\\":
                i += 1
            elif c == "'":
                in_sq = False
        elif in_dq:
            if c == "\\":
                i += 1
            elif c == '"':
                in_dq = False
        else:
            if c == "/" and i + 1 < n:
                if src[i + 1] == "/":
                    in_sl = True
                    i += 1
                elif src[i + 1] == "*":
                    in_ml = True
                    i += 1
            elif c == "'":
                in_sq = True
            elif c == '"':
                in_dq = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    # main body is from first char after '{' to this '}' (exclusive)
                    lb = m.end()  # first char after '{'
                    rb = i  # position of matching '}'
                    return (lb, rb)
        i += 1
    return (None, None)


def _find_last_toplevel_return(src: str, lb: int, rb: int):
    """Return absolute index of the last top-level `return ...;` inside main body, or None."""
    i = lb
    depth = 1  # we're just after '{' of main(), so depth starts at 1
    n = len(src)
    in_sq = in_dq = in_sl = in_ml = False
    last = None
    while i < rb:
        c = src[i]
        if in_sl:
            if c == "\n":
                in_sl = False
        elif in_ml:
            if c == "*" and i + 1 < n and src[i + 1] == "/":
                in_ml = False
                i += 1
        elif in_sq:
            if c == "\\":
                i += 1
            elif c == "'":
                in_sq = False
        elif in_dq:
            if c == "\\":
                i += 1
            elif c == '"':
                in_dq = False
        else:
            if c == "/" and i + 1 < n:
                if src[i + 1] == "/":
                    in_sl = True
                    i += 1
                elif src[i + 1] == "*":
                    in_ml = True
                    i += 1
            elif c == "'":
                in_sq = True
            elif c == '"':
                in_dq = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif (
                depth == 1
                and src.startswith("return", i)
                and (i + 6 >= n or not (src[i + 6].isalnum() or src[i + 6] == "_"))
            ):
                last = i
        i += 1
    return last


def ensure_dump_call(src: str) -> str:
    if not HAS_MAIN_RE.search(src):
        return src
    if "SYCL_TIMER_DUMP(" in src:
        return src

    lb, rb = _find_main_span(src)
    if lb is None:
        return src

    # prefer last top-level return in main(); if none, put before closing brace
    pos = _find_last_toplevel_return(src, lb, rb)
    if pos is not None:
        insert_at = pos
        return src[:insert_at] + "SYCL_TIMER_DUMP();\n" + src[insert_at:]
    else:
        return src[:rb] + "\n  SYCL_TIMER_DUMP();\n" + src[rb:]


"""
def ensure_dump_call(src: str) -> str:
    # Only if file defines main()
    if not HAS_MAIN_RE.search(src):
        return src
    if 'SYCL_TIMER_DUMP(' in src:
        return src

    lb, rb = _find_main_span(src)
    if lb is None:
        return src

    main_body = src[lb:rb]

    # If any return exists inside main, inject dump *before the last one*
    last_ret = None
    for mm in RETURN_ANY_RE.finditer(main_body):
        last_ret = mm
    if last_ret:
        insert_at = lb + last_ret.start()
        return src[:insert_at] + 'SYCL_TIMER_DUMP();\n' + src[insert_at:]

    # Otherwise, add just before closing brace of main
    return src[:rb] + '\n  SYCL_TIMER_DUMP();\n' + src[rb:]
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: instrument_sycl_timers.py <repo_root>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"Path not found: {root}")
        sys.exit(1)

    # Write header if missing
    hdr = root / HEADER_NAME
    if not hdr.exists():
        hdr.write_text(SYCL_TIMER_HPP, encoding="utf-8")
        print(f"[+] Wrote {hdr.relative_to(root)}")

    changed = 0
    scanned = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == HEADER_NAME:
            continue
        if p.suffix.lower() not in SOURCE_EXTS:
            continue
        if any(
            part in p.parts for part in (".git", "build", "cmake-build", "CMakeFiles")
        ):
            continue
        if p.name.endswith(".orig"):
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        scanned += 1
        new_text = instrument_source(text, p)
        needs_timer = any(
            s in new_text
            for s in (
                "SYCL_TIME_AGG(",
                "SYCL_TIME_EVT(",
                "SYCL_TIME(",
                "SYCL_TIMER_DUMP(",
            )
        )
        if (
            needs_timer
            and "sycl_timer.hpp" not in new_text
            and p.suffix.lower() != ".sycl"
        ):
            new_text = ensure_timer_include(new_text, p)
        if new_text != text and write_or_skip(p, new_text):
            changed += 1
            print(f"[mod] {p.relative_to(root)}")

    print(f"\nDone. Scanned {scanned} files; modified {changed}.")
    print(
        "Build normally. You will see only aggregate kernel timings (sum & avg) in terminal and sycl_timing.csv."
    )


if __name__ == "__main__":
    main()
