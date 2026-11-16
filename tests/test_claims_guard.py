"""Tests for the guard a claims-task program runs under: what its process may read, write
and start."""

import tempfile
import unittest
from pathlib import Path

from runsleuth import claims_run

# What the process does is chosen by its first argument; the rest are paths.
SCRIPT = """
import os
import sys

action, paths = sys.argv[1], sys.argv[2:]
if action == "read":
    print("read", len(open(paths[0], encoding="utf-8").read()))
elif action == "write":
    with open(paths[0], "w", encoding="utf-8") as stream:
        stream.write("written")
    print("wrote")
elif action == "list":
    print("listed", len(os.listdir(paths[0])))
elif action == "walk":
    print("walked", sum(len(files) for _, _, files in os.walk(paths[0], onerror=print)))
elif action == "remove":
    os.remove(paths[0])
elif action == "rename":
    os.replace(paths[0], paths[1])
elif action == "temporary":
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt") as stream:
        stream.write("x")
    print("temporary", tempfile.gettempdir())
elif action == "hash":
    print("hash", hash("a fixed string"))
elif action == "environment":
    print("environment", sorted(name for name in os.environ if "SECRET" in name))
elif action == "process":
    import subprocess

    subprocess.run([sys.executable, "-c", "print('started')"], check=True)
elif action == "worker":
    import multiprocessing

    def look(path, queue):
        try:
            open(path, encoding="utf-8").read()
            queue.put("read")
        except PermissionError as error:
            queue.put(str(error))

    if __name__ == "__main__":
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        worker = context.Process(target=look, args=(paths[0], queue))
        worker.start()
        print("worker", queue.get(timeout=60))
        worker.join()
"""


class GuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ("data", "out", "secret", "program"):
            (self.root / name).mkdir()
        (self.root / "data" / "table.csv").write_text("row,label\n0,1\n", encoding="utf-8")
        (self.root / "secret" / "labels.csv").write_text("row,label\n0,0\n", encoding="utf-8")
        self.program = self.root / "program" / "train.py"
        self.program.write_text(SCRIPT, encoding="utf-8")

    def call(self, *arguments):
        return claims_run.call_program(
            self.program,
            [str(argument) for argument in arguments],
            self.root / "out",
            timeout=120,
            read=[self.root / "data"],
            write=[self.root / "out"],
        )

    def refused(self, record, verb):
        self.assertEqual(record["status"], "crashed", record["output"])
        self.assertIn("PermissionError", record["output"])
        self.assertIn(f"outside what this program may {verb}", record["output"])

    def test_the_process_reads_its_data_and_its_output_and_nothing_beside_them(self):
        allowed = self.call("read", self.root / "data" / "table.csv")
        self.assertEqual((allowed["status"], allowed["output"].strip()), ("completed", "read 14"))
        self.refused(self.call("read", self.root / "secret" / "labels.csv"), "read")
        self.refused(self.call("read", self.root / "data" / ".." / "secret" / "labels.csv"), "read")
        self.assertEqual(self.call("list", self.root / "data")["output"].strip(), "listed 1")
        self.refused(self.call("list", self.root), "read")
        self.refused(self.call("list", self.root / "secret"), "read")
        # A walk cannot list what it may not read: it finds no file there.
        walked = self.call("walk", self.root / "secret")
        self.assertIn("walked 0", walked["output"])
        # The program may read itself, not what lies beside it.
        (self.root / "program" / "attempt.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.call("read", self.program)["status"], "completed")
        self.refused(self.call("read", self.root / "program" / "attempt.json"), "read")

    def test_the_process_writes_its_output_only(self):
        wrote = self.call("write", self.root / "out" / "report.json")
        self.assertEqual(wrote["status"], "completed", wrote["output"])
        self.assertEqual((self.root / "out" / "report.json").read_text(), "written")
        self.assertEqual(
            self.call("read", self.root / "out" / "report.json")["status"], "completed"
        )
        self.refused(self.call("write", self.root / "data" / "table.csv"), "write")
        self.refused(self.call("write", self.root / "elsewhere.txt"), "write")
        self.refused(self.call("remove", self.root / "data" / "table.csv"), "write")
        self.refused(
            self.call("rename", self.root / "out" / "report.json", self.root / "moved.json"),
            "write",
        )
        self.assertTrue((self.root / "data" / "table.csv").is_file())
        self.assertFalse((self.root / "elsewhere.txt").exists())

    def test_the_process_has_temporary_files_a_fixed_hash_seed_and_no_credentials(self):
        temporary = self.call("temporary")
        self.assertEqual(temporary["status"], "completed", temporary["output"])
        where = Path(temporary["output"].split("temporary ", 1)[1].strip())
        self.assertFalse(where.exists())
        hashes = {self.call("hash")["output"].strip() for _ in range(2)}
        self.assertEqual(len(hashes), 1)
        import os

        os.environ["RUNSLEUTH_TEST_SECRET"] = "x"
        self.addCleanup(os.environ.pop, "RUNSLEUTH_TEST_SECRET")
        self.assertEqual(self.call("environment")["output"].strip(), "environment []")

    def test_the_process_starts_its_own_workers_and_no_other_process(self):
        started = self.call("process")
        self.assertEqual(started["status"], "crashed")
        self.assertIn("may not", started["output"])
        self.assertNotIn("started", started["output"].splitlines())
        allowed = self.call("worker", self.root / "data" / "table.csv")
        self.assertEqual(allowed["output"].strip().splitlines()[-1], "worker read", allowed)
        # A worker runs under the same guard.
        refused = self.call("worker", self.root / "secret" / "labels.csv")
        self.assertIn("outside what this program may read", refused["output"])

    def test_the_guard_is_asleep_without_its_configuration(self):
        import subprocess
        import sys

        environment = claims_run.guarded_environment([], [], self.root / "out")
        del environment["RUNSLEUTH_GUARD"]
        completed = subprocess.run(
            [sys.executable, str(self.program), "read", str(self.root / "secret" / "labels.csv")],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.stdout.strip(), "read 14")


if __name__ == "__main__":
    unittest.main()
