from typing import List, Optional
import requests
import json
import threading


def run_coverage_batched(server, codes, tests, timeout=60, timeout_on_client=False) -> List[int]:
    """
    Executes a batch of code snippets and tests them with a set of tests,
    then returns the coverage percentage using coverage.py.
    """
    threads = []
    results: List[Optional[int]] = [None] * len(codes)

    def run_coverage_threaded(i, code, test):
        data = json.dumps({"code": code + "\n\n" + "\n".join(test), "timeout": timeout})
        try:
            r = requests.post(
                server + "/py_coverage",
                data=data,
                timeout=(timeout + 20) if timeout_on_client else None
            )
            results[i] = int(r.text)
        except Exception as e:
            results[i] = -3  # In case of an error during the coverage check
            print(f"Error executing coverage for test {i}: {str(e)}")

    for i, (code, test) in enumerate(zip(codes, tests)):
        t = threading.Thread(target=run_coverage_threaded, args=(i, code, test))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    
    results_new = []
    for r in results:
        if r is None:
            results_new.append(-3)  
        else:
            results_new.append(r)

    return results_new
