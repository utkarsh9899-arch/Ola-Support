"""Offline guard: proves the project makes no outbound network calls.

Put this directory on PYTHONPATH and Python imports it at start-up (sitecustomize). Every DNS lookup or
socket connection to anything other than localhost is BLOCKED and recorded in $OFFLINE_GUARD_LOG.

    PYTHONPATH=tools/offline_guard OFFLINE_GUARD_LOG=offline_guard.log python demos/run_all_demos.py
"""
import os
import socket

_LOG = os.environ.get("OFFLINE_GUARD_LOG")
_LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def _record(kind: str, target) -> None:
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"BLOCKED {kind} {target!r} pid={os.getpid()}\n")


_orig_getaddrinfo = socket.getaddrinfo
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _host(address) -> str:
    return str(address[0]) if isinstance(address, tuple) else str(address)


def getaddrinfo(host, *args, **kwargs):
    if host is not None and str(host) not in _LOCAL:
        _record("dns", host)
        raise OSError(f"offline guard: blocked DNS lookup for {host!r}")
    return _orig_getaddrinfo(host, *args, **kwargs)


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and _host(address) not in _LOCAL:
        _record("connect", address)
        raise OSError(f"offline guard: blocked connection to {address!r}")
    return _orig_connect(self, address)


def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and _host(address) not in _LOCAL:
        _record("connect_ex", address)
        return 111  # ECONNREFUSED
    return _orig_connect_ex(self, address)


socket.getaddrinfo = getaddrinfo
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
if _LOG:
    with open(_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"guard active pid={os.getpid()}\n")
