"""Limits what the interpreter of a claims-task program may touch.

claims_run puts this directory on PYTHONPATH, so the interpreter imports this file as
sitecustomize when it starts, in the program's process and in every worker the program
starts. RUNSLEUTH_GUARD holds, as JSON, the files and directories the process may read
and those it may write. An audit hook then refuses every other file or directory the
process opens, lists, enters, creates, renames or removes; every process it starts other
than a worker of its own interpreter; and every network connection.

This keeps a program from reading what lies beside its data, such as sealed labels. It is
not a sandbox: code that loads a foreign library can get around an audit hook, and the
screen of the program's source (standalone_check.screen) refuses the imports that would.
Whether a path exists can still be found out. Only the standard library is used here.
"""

import json
import os
import sys

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_DEVICES = ("nul", "con", "conin$", "conout$")
# Events whose named arguments are paths the process must be allowed to read.
_READS = {"os.listdir": (0,), "os.scandir": (0,), "os.chdir": (0,)}
# Events whose named arguments are paths the process must be allowed to write.
_WRITES = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.rename": (0, 1),
    "os.link": (0, 1),
    "os.symlink": (0, 1),
}
_REFUSED = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.startfile",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.getaddrinfo",
    "urllib.Request",
    "http.client.connect",
    "ftplib.connect",
    "smtplib.connect",
    "webbrowser.open",
)


def _normal(path):
    return os.path.normcase(os.path.realpath(os.fsdecode(path)))


def install(configuration):
    """Add the audit hook for a configuration {"read": [...], "write": [...]}."""
    write = tuple(_normal(path) for path in configuration["write"])
    read = tuple(_normal(path) for path in configuration["read"]) + write
    interpreter = _normal(sys.executable)

    def check(path, roots, event, verb):
        if path is None:
            path = os.getcwd()
        if isinstance(path, int):
            return
        try:
            text = os.fsdecode(path)
        except TypeError:
            return
        if text.lower() in _DEVICES:
            return
        found = _normal(text)
        if found.startswith("\\\\.\\"):
            return
        for root in roots:
            if found == root or found.startswith(root + os.sep):
                return
        raise PermissionError(f"{event}: {text} is outside what this program may {verb}")

    def hook(event, arguments):
        if event == "open":
            path, mode, flags = arguments[0], arguments[1], arguments[2]
            if isinstance(mode, str):
                writing = any(letter in mode for letter in "wax+")
            else:
                writing = bool((flags or 0) & _WRITE_FLAGS)
            if writing:
                check(path, write, event, "write")
            else:
                check(path, read, event, "read")
        elif event in _READS:
            for position in _READS[event]:
                check(arguments[position], read, event, "read")
        elif event in _WRITES:
            for position in _WRITES[event]:
                check(arguments[position], write, event, "write")
        elif event == "_winapi.CreateProcess":
            # A worker of the program's own interpreter, as a data loader starts them.
            if not arguments[0] or _normal(arguments[0]) != interpreter:
                raise PermissionError(f"{event}: this program may not start a process")
        elif event in _REFUSED:
            raise PermissionError(f"{event}: this program may not do this")

    sys.addaudithook(hook)


if os.environ.get("RUNSLEUTH_GUARD"):
    install(json.loads(os.environ["RUNSLEUTH_GUARD"]))
