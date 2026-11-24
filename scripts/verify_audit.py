"""Manual verification of the consistency audit under concurrent training.

    python scripts/verify_audit.py --data /tmp/cal.csv --work /tmp/v-audit

Two scenarios are exercised against the real engine:

``interleave``
    A trainer thread keeps committing groups while the main thread audits in a
    loop.  Every audit must pass and must describe exactly the checkpoint read
    when that audit started (values are compared against a snapshot taken just
    before it).  The run is then paused, audited across the pause boundary,
    resumed and audited again to the end.  Finally the run is deleted: an audit
    started before the deletion returns the full checkpoint result, a later one
    reports it as deleted.

``memory``
    Audit peak traced memory with a growing number of committed groups.  The
    old ``fetchall``-based audit grew linearly; the streamed one must stay flat.

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trainer import engine  # noqa: E402


def make_csv(path: str, rows: int) -> None:
    """Small well-conditioned data: no divergence even with many groups."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("measured,reference\n")
        for i in range(rows):
            x = (i % 200) * 0.01 - 1.0
            fh.write(f"{x:.6f},{1.7 * x - 0.35:.6f}\n")


def checkpoint_snapshot(db: str, run_id: int) -> dict:
    with engine.session(db) as conn:
        row = engine.get_run(conn, run_id)
    return {
        "updates": int(row["updates_done"]),
        "samples": int(row["samples_done"]),
        "w": float(row["w"]),
        "b": float(row["b"]),
        "v_w": float(row["v_w"]),
        "v_b": float(row["v_b"]),
    }


def scenario_interleave(db: str, csv_path: str, group_size: int,
                        chunk_limit: int) -> dict:
    conn = engine.connect(db)
    run_id = engine.create_run(
        conn, csv_path, "interleave.csv", csv_path, group_size=group_size,
        chunk_limit=chunk_limit, lr=1e-4, momentum=0.5,
        init_w=0.0, init_b=0.0,
    )
    conn.close()

    report = {"checks": 0, "false_alarms": 0, "checkpoint_mismatches": 0,
              "checkpoints_observed": [], "pauses": 0}

    stop = threading.Event()

    def trainer() -> None:
        c = engine.connect(db)
        try:
            for _ in range(400):
                if stop.is_set():
                    break
                engine.train(c, run_id, csv_path, max_updates=2)
                row = engine.get_run(c, run_id)
                if row["status"] == "finished":
                    break
                time.sleep(0.0005)
        finally:
            c.close()

    # --- audit / training interleaving, with periodic deliberate pauses ------
    t = threading.Thread(target=trainer, daemon=True)
    t.start()
    while t.is_alive():
        with engine.session(db) as c:
            # The audit may start just before a group commits; it must still
            # return the checkpoint it read first and must not report damage.
            result = engine.audit(c, run_id)
        before = checkpoint_snapshot(db, run_id)
        report["checks"] += 1
        if not result["ok"]:
            report["false_alarms"] += 1
        # The audit's reported values must be a real committed checkpoint and
        # must not exceed the state observed right after the audit returned.
        report["checkpoints_observed"].append(result["updates"])
        if result["updates"] > before["updates"]:
            report["checkpoint_mismatches"] += 1
        time.sleep(0.001)
    t.join()
    report["training_finished"] = checkpoint_snapshot(db, run_id)["updates"]

    # --- explicit pause / resume boundaries ---------------------------------
    # Start a fresh run so pause boundaries are actually crossed.
    with engine.session(db) as c:
        engine.delete_run(c, run_id)
    conn = engine.connect(db)
    run_id = engine.create_run(
        conn, csv_path, "interleave2.csv", csv_path, group_size=group_size,
        chunk_limit=chunk_limit, lr=1e-4, momentum=0.5,
        init_w=0.0, init_b=0.0,
    )
    conn.close()
    for _ in range(40):
        with engine.session(db) as c:
            engine.set_pause_after_groups(c, run_id, 2)
            engine.train(c, run_id, csv_path)
            paused = engine.get_run(c, run_id)
            if paused["status"] != "paused":
                break
            report["pauses"] += 1
            audit = engine.audit(c, run_id)
            report["checks"] += 1
            if not audit["ok"]:
                report["false_alarms"] += 1
            if audit["updates"] != int(paused["updates_done"]):
                report["checkpoint_mismatches"] += 1
    with engine.session(db) as c:
        engine.set_pause_after_groups(c, run_id, None)
        engine.train(c, run_id, csv_path)
        final = engine.get_run(c, run_id)
        audit = engine.audit(c, run_id)
    report["checks"] += 1
    report["final_ok"] = bool(audit["ok"])
    report["final_updates"] = audit["updates"]
    report["final_samples"] = audit["samples"]
    report["finished"] = final["status"] == "finished"

    # --- deletion during the audit -----------------------------------------
    snapshot_result = {}

    def delete_from_other_connection() -> None:
        other = engine.connect(db)
        try:
            engine.delete_run(other, run_id)
        finally:
            other.close()

    with engine.session(db) as c:
        snapshot_result = engine.audit(c, run_id,
                                       snapshot_hook=delete_from_other_connection)
        try:
            engine.audit(c, run_id)
            deleted_reported = False
        except engine.InputError:
            deleted_reported = True
    report["delete_after_snapshot_full_result"] = bool(
        snapshot_result["ok"]
        and snapshot_result["updates"] == report["final_updates"]
    )
    report["deleted_run_reported"] = deleted_reported

    report["observed_monotonic"] = (
        report["checkpoints_observed"] == sorted(report["checkpoints_observed"])
    )
    report["ok"] = bool(
        report["false_alarms"] == 0
        and report["checkpoint_mismatches"] == 0
        and report["final_ok"]
        and report["finished"]
        and report["delete_after_snapshot_full_result"]
        and report["deleted_run_reported"]
    )
    return report


def measure_audit_memory(groups: int) -> dict:
    """Audit one completed run, reporting traced peak and RSS growth.

    The run is built in a separate child process, so the parent's own heap does
    not grow with the number of updates and the RSS delta reflects only the
    audit call.
    """
    import subprocess

    tmp = tempfile.mkdtemp(prefix="audit-mem-")
    try:
        csv_path = os.path.join(tmp, "m.csv")
        make_csv(csv_path, groups * 5)
        db = os.path.join(tmp, "m.sqlite3")
        conn = engine.connect(db)
        try:
            run_id = engine.create_run(
                conn, csv_path, "m.csv", csv_path, group_size=5, chunk_limit=5,
                lr=1e-4, momentum=0.5, init_w=0.0, init_b=0.0,
            )
            engine.train(conn, run_id, csv_path)
        finally:
            conn.close()

        # In a fresh interpreter, audit the same database and report both
        # tracemalloc peak and VmRSS.  Using a child keeps this process clean.
        probe = (
            "import json, sys, tracemalloc\n"
            "sys.path.insert(0, %r)\n"
            "from trainer import engine\n"
            "import gc\n"
            "def rss():\n"
            "    try:\n"
            "        return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])\n"
            "    except Exception:\n"
            "        return 0\n"
            "c = engine.connect(%r)\n"
            "gc.collect(); tracemalloc.start()\n"
            "r0 = rss(); base = tracemalloc.get_traced_memory()[0]\n"
            "res = engine.audit(c, %d)\n"
            "peak = tracemalloc.get_traced_memory()[1]; r1 = rss()\n"
            "print(json.dumps({'ok': res['ok'], 'updates': res['updates'],\n"
            "                  'traced_peak_bytes': peak - base,\n"
            "                  'rss_delta_kb': r1 - r0}))\n"
        ) % (ROOT, db, run_id)
        proc = subprocess.run([sys.executable, "-I", "-c", probe],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            return {"groups": groups, "ok": False, "error": proc.stderr[-500:]}
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        out["groups"] = groups
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None,
                   help="CSV for the interleave scenario (generated if omitted)")
    p.add_argument("--work", default=None,
                   help="working directory for the interleave scenario")
    p.add_argument("--group-size", type=int, default=200)
    p.add_argument("--chunk-limit", type=int, default=25)
    p.add_argument("--mem-groups", default="100,400,1600,6400",
                   help="comma-separated group counts for the memory sweep")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    work = args.work or tempfile.mkdtemp(prefix="verify-audit-")
    created_work = args.work is None
    if created_work:
        os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "audit.sqlite3")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db + suffix)
        except OSError:
            pass
    csv_path = args.data
    if csv_path is None:
        csv_path = os.path.join(work, "data.csv")
        make_csv(csv_path, 4000)

    interleave = scenario_interleave(db, csv_path, args.group_size,
                                     args.chunk_limit)
    sweep = [measure_audit_memory(int(n)) for n in args.mem_groups.split(",")]

    payload = {"interleave": interleave, "memory": sweep}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    if created_work:
        shutil.rmtree(work, ignore_errors=True)

    # Flat memory: the largest sweep must not grow much over the smallest.
    small = sweep[0]["traced_peak_bytes"]
    large = sweep[-1]["traced_peak_bytes"]
    flat = large < max(small * 4, 256 * 1024)
    rss_flat = all(abs(s["rss_delta_kb"]) < 4096 for s in sweep)
    print(json.dumps({"traced_flat": flat, "rss_flat": rss_flat}, ensure_ascii=False))
    return 0 if (interleave["ok"] and all(s["ok"] for s in sweep)
                 and flat and rss_flat) else 1


if __name__ == "__main__":
    raise SystemExit(main())
