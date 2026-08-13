"""Inspect Python 3.6 bytecode without importing or executing the module.

Run this utility with the legacy ``ta-gan`` environment. Python bytecode is
version-specific, so Python 3.8 cannot reliably unmarshal the retained 3.6
cache files.
"""

from __future__ import print_function

import argparse
import hashlib
import marshal
import os
import sys
import types


PY36_HEADER_SIZE = 12


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def walk_code(code, prefix=""):
    """Print stable metadata for a code object and all nested code objects."""
    print(
        "{}{} args={} bytecode={} consts={}".format(
            prefix,
            code.co_name,
            code.co_argcount,
            sha256(code.co_code)[:16],
            len(code.co_consts),
        )
    )
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            walk_code(constant, prefix + "  ")


def load_code(path):
    with open(path, "rb") as stream:
        data = stream.read()
    return data, marshal.loads(data[PY36_HEADER_SIZE:])


def main(paths):
    if sys.version_info[:2] != (3, 6):
        raise RuntimeError("Run with Python 3.6; found {}".format(sys.version))

    for path in paths:
        data, code = load_code(path)
        print(
            "FILE {} bytes={} sha256={}".format(
                os.path.normpath(path), len(data), sha256(data)
            )
        )
        walk_code(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    main(parser.parse_args().paths)
