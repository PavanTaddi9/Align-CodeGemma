from typing import List, Optional, Tuple
import requests
import json
import threading


def run_coverage(server, code, tests, timeout=60, timeout_on_client=False) -> int:
    """
    Executes a code snippet and tests it with a set of tests,
    then returns the coverage percentage using coverage.py.
    """
    tests_str = "\n".join(tests)
    code_with_tests = code + "\n\n" + tests_str
    data = json.dumps({"code": code_with_tests, "timeout": timeout})
    try:
        r = requests.post(
            server + "/py_coverage",
            data=data,
            timeout=(timeout + 20) if timeout_on_client else None
        )
        return int(r.text)
    except Exception as e:
        print(e)
        return -3


def run_coverage_batched(server, codes, tests, timeout=60, timeout_on_client=False) -> List[int]:
    """
    Executes a batch of code snippets and tests them with a set of tests,
    then returns the coverage percentage using coverage.py.
    """
    threads = []
    results: List[Optional[int]] = [None] * len(codes)

    def run_coverage_threaded(i, code, test):
        results[i] = run_coverage(
            server, code, test, timeout, timeout_on_client)

    for i, (code, test) in enumerate(zip(codes, tests)):
        t = threading.Thread(target=run_coverage_threaded,
                             args=(i, code, test))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    results_new = []
    for r in results:
        if r is None:
            results_new.append((False, "Failed to execute program"))
        else:
            results_new.append(r)

    return results_new