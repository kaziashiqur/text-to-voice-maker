"""Internet calls that the Cancel button can stop at any moment.

Each request runs in a helper thread; we wait in small steps and stop waiting as soon as
Cancel is pressed. A live seconds counter shows that a slow service is still working.
"""
import threading
import time

import requests


class Cancelled(Exception):
    pass


_cancel: threading.Event | None = None


def set_cancel(event: threading.Event | None) -> None:
    global _cancel
    _cancel = event


def check() -> None:
    if _cancel is not None and _cancel.is_set():
        raise Cancelled()


def run(fn, notify=None, label: str = ""):
    check()
    box = {}

    def target():
        try:
            box["r"] = fn()
        except BaseException as e:      # handed back to the caller below
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    start, shown = time.time(), 0
    while t.is_alive():
        t.join(0.25)
        check()
        secs = int(time.time() - start)
        if notify and label and secs >= 2 and secs != shown:
            shown = secs
            notify(f"{label} {secs} s")
    if "e" in box:
        raise box["e"]
    return box.get("r")


def post(url, notify=None, label: str = "", **kw):
    return run(lambda: requests.post(url, **kw), notify, label)


def get(url, notify=None, label: str = "", **kw):
    return run(lambda: requests.get(url, **kw), notify, label)


def sleep(seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        check()
        time.sleep(0.2)
