#!/usr/bin/env python3
r"""
pack_py_files_rel.py

Bundles .py files from given paths into one output file. Each file is written as:

# path: relative/path/to/file.py
<file contents>

# end of file: relative/path/to/file.py

By default relative paths are computed relative to the output file's directory.
Use --relative-to cwd to compute relative paths from the current working directory instead.

For DQN investigation : 
python copy_paste.py copy.txt debug\dqn_1_env_cartpole.py rl_agents\service.py rl_agents\agent.py rl_agents\value_functions rl_agents\value_agents rl_agents\utils rl_agents\replay_memory\replay_memory.py rl_agents\replay_memory\sampler.py rl_agents\policies\epsilon_greedy_proxy.py
"""

from pathlib import Path
import argparse
import os
from typing import Iterable, List, Set


def gather_py_files(paths: Iterable[Path], skip_paths: Set[Path]) -> List[Path]:
    found = []
    seen = set()
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() == ".py":
                resolved = p.resolve()
                if resolved not in seen and resolved not in skip_paths:
                    seen.add(resolved)
                    found.append(resolved)
        else:
            for root, dirs, files in os.walk(p):
                for fname in files:
                    if fname.lower().endswith(".py"):
                        fpath = (Path(root) / fname).resolve()
                        if fpath in seen or fpath in skip_paths:
                            continue
                        seen.add(fpath)
                        found.append(fpath)
    found.sort()
    return found


def write_bundle(files: List[Path], out_path: Path, relative_to: str) -> None:
    if relative_to == "output":
        base = out_path.parent.resolve()
    else:
        base = Path.cwd().resolve()

    with out_path.open("w", encoding="utf-8", errors="replace", newline="\n") as out:
        for i, fpath in enumerate(files, start=1):
            # compute relative path, fallback to absolute if relpath fails (different drives on Windows)
            try:
                rel = os.path.relpath(str(fpath), start=str(base))
            except Exception:
                rel = str(fpath)
            out.write(f"# path: {rel}\n")
            try:
                with fpath.open("r", encoding="utf-8", errors="replace") as fh:
                    out.write(fh.read())
            except Exception as e:
                out.write(f"# ERROR reading file: {e}\n")
            out.write("\n\n")
            out.write(f"# end of file: {rel}\n")
            if i != len(files):
                out.write("\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Bundle .py files into one file with relative paths.")
    ap.add_argument(
        "--relative-to",
        choices=("output", "cwd"),
        default="cwd",
        help="Make paths relative to the output file's directory (default) or to current working directory.",
    )
    ap.add_argument("output", type=Path, help="Output file to write the bundle to.")
    ap.add_argument("paths", type=Path, nargs="+", help="Files or directories to collect .py files from.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path: Path = args.output.resolve()
    input_paths = [p.resolve() for p in args.paths]

    # avoid including the output file itself if it's inside the inputs
    skip_paths = {out_path}

    py_files = gather_py_files(input_paths, skip_paths)
    if not py_files:
        print("No .py files found in provided paths. Nothing written.")
        return

    write_bundle(py_files, out_path, args.relative_to)
    print(f"Wrote {len(py_files)} .py files to {out_path} (paths relative to {args.relative_to})")


if __name__ == "__main__":
    main()

