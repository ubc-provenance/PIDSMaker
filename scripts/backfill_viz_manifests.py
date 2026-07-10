#!/usr/bin/env python3
"""Backfill viz_manifest.json for eval runs that predate manifest writing.

The Run Browser only lists a run if its eval dir has a viz_manifest.json (or
viz/*_points.json). New evaluations write the manifest automatically; older runs
have none, so they stay hidden. This calls the pipeline's own write_viz_manifest
on each completed run that lacks one -- it only indexes artifacts already on disk,
nothing is retrained. Idempotent (skips existing unless --force).

Run inside the container:
    python -m scripts.backfill_viz_manifests [--dry-run] [--force] [--artifacts-root PATH]
"""

import argparse
import glob
import os
import sys

from pidsmaker.config.pipeline import (
    TASK_FINISHED_FILE,
    VIZ_MANIFEST_FILE,
    write_viz_manifest,
)


def resolve_artifacts_root(override=None):
    """Mirror viz_server.get_artifacts_root so we scan what the server reads."""
    if override:
        return override
    if os.path.isdir("/home/artifacts"):
        return "/home/artifacts"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("PIDS_ARTIFACTS_DIR", os.path.join(repo_root, "artifacts"))


def find_eval_runs(artifacts_root):
    """Same patterns the Run Browser discovers from."""
    patterns = [
        os.path.join(artifacts_root, "evaluation/evaluation/*/*"),
        os.path.join(artifacts_root, "detection/evaluation/*/*"),
    ]
    runs = []
    for pat in patterns:
        runs.extend(d for d in glob.glob(pat) if os.path.isdir(d))
    return sorted(set(runs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", default=None,
                    help="Override the artifacts root (defaults to the server's resolution).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written without writing anything.")
    ap.add_argument("--force", action="store_true",
                    help="Rewrite manifests that already exist.")
    args = ap.parse_args()

    root = resolve_artifacts_root(args.artifacts_root)
    if not os.path.isdir(root):
        print(f"ERROR: artifacts root does not exist: {root}", file=sys.stderr)
        return 1
    print(f"[backfill] artifacts root: {root}")

    runs = find_eval_runs(root)
    print(f"[backfill] found {len(runs)} eval run dir(s)")

    written = skipped_have = skipped_incomplete = 0
    for d in runs:
        has_manifest = os.path.exists(os.path.join(d, VIZ_MANIFEST_FILE))
        completed = os.path.exists(os.path.join(d, TASK_FINISHED_FILE))
        has_pr = os.path.isdir(os.path.join(d, "precision_recall_dir"))

        if has_manifest and not args.force:
            skipped_have += 1
            continue
        # write_viz_manifest no-ops without a precision_recall_dir; report it here
        # so incomplete runs are visibly distinguished from backfilled ones.
        if not has_pr:
            skipped_incomplete += 1
            print(f"  SKIP (no precision_recall_dir, completed={completed}): {d}")
            continue

        if args.dry_run:
            print(f"  WOULD WRITE: {os.path.join(d, VIZ_MANIFEST_FILE)}")
            written += 1
            continue

        write_viz_manifest(d)
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"\n[backfill] {verb} {written} manifest(s); "
          f"{skipped_have} already had one; "
          f"{skipped_incomplete} skipped (incomplete, no precision_recall_dir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
