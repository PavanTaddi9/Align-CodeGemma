from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import threading


def exec_test(server, code, timeout=30, timeout_on_client=False, stdin="") -> Tuple[bool, str]:
    """
    Executes a test against a code snippet.
    Produces true if the test passes, false otherwise.
    Also returns the output of the code (sterr if it fails, stdout if it passes).

    You can set test to an empty string if you want to execute the code without any tests
    and just check if it runs without errors.

    timeout_on_client: If true, the client will timeout after timeout+2 seconds.
    """
    code_with_tests = code + "\n\n" 
    data = json.dumps(
        {"code": code_with_tests, "timeout": timeout, "stdin": stdin})
    try:
        r = requests.post(
            server + "/py_exec",
            data=data,
            timeout=(timeout + 2) if timeout_on_client else None
        )
        lines = r.text.split("\n")
        resp = lines[0]
        outs = "\n".join(lines[1:])
        assert resp == "0" or resp == "1"
        return resp == "0", outs
    except Exception as e:
        return False, "Failed to execute program: " + str(e)



def exec_test_batched(server, codes, lang=None, timeout=30, timeout_on_client=False, stdins=None) -> List[int]:
    stdins = stdins or [None] * len(codes)

    def exec_fn(code_stdin):
        code, stdin = code_stdin
        try:
            success, _ = exec_test(server, code, timeout, timeout_on_client, stdin=stdin)
            return 1 if success else -1
        except:
            return -1

    with ThreadPoolExecutor(max_workers=min(32, len(codes))) as executor:
        results = list(executor.map(exec_fn, zip(codes, stdins)))

    return results


