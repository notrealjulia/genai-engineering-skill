"""Code Execution: the LLM writes a Python script, a separate runtime executes it, errors
are fed back for self-correction, and the output grounds the final answer.

Based on the pattern described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book has
no code for this pattern.

WARNING: run_python() is a *subprocess with limits*, NOT a security sandbox. It stops
runaway loops and stray file writes in the working dir, not a hostile script. In
production pass `runner=` a real sandbox (container/microVM with no network, read-only
filesystem, CPU/memory limits) or use your provider's hosted code-execution tool.
"""
import os
import re
import subprocess
import sys
import tempfile

from llm_client import call_llm

CODE_PROMPT = """Write a complete Python 3 script that solves the task below.
Use only the standard library. Print the final result to stdout.
Reply with only the code in a ```python block.

Task: {task}"""

FIX_PROMPT = """Your script failed.

```python
{code}
```

Error:
{error}

Fix it. Reply with only the corrected code in a ```python block."""


def extract_code(raw: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw or "", re.DOTALL)
    return blocks[0].strip() if blocks else None


def run_python(code: str, timeout: float = 10.0, max_output: int = 4000) -> dict:
    with tempfile.TemporaryDirectory() as workdir:  # scratch dir, deleted afterwards
        env = {"PYTHONIOENCODING": "utf-8"}  # no inherited secrets (API keys) in the env
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")  # Python needs it on Windows
        limits = None
        if os.name == "posix":
            def limits():  # CPU seconds and memory, so `while True` can't eat the host
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (int(timeout), int(timeout)))
                resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=workdir, env=env,
                               capture_output=True, text=True, timeout=timeout,
                               preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"Timed out after {timeout}s"}
    return {"ok": p.returncode == 0, "stdout": p.stdout[-max_output:],
            "stderr": p.stderr[-max_output:]}


def code_execution(task: str, max_attempts: int = 3, runner=run_python) -> dict:
    attempts, raw = [], call_llm(CODE_PROMPT.format(task=task))
    for attempt in range(max_attempts):
        if attempt:  # self-correction: give the model its own code and the real error
            raw = call_llm(FIX_PROMPT.format(code=code or raw[:2000], error=error))
        code = extract_code(raw)
        if code is None:
            error = "No ```python code block found in your reply."
            attempts.append({"code": None, "error": error})
        else:
            result = runner(code)
            if result["ok"] and result["stdout"].strip():
                attempts.append({"code": code, "output": result["stdout"]})
                # Code that runs can still be wrong (bad join, wrong formula): the answer
                # shows the code so a human or a check can audit it.
                answer = call_llm(f"Task: {task}\n\nThis code was run:\n{code}\n\nOutput:\n"
                                  f"{result['stdout']}\n\nAnswer the task using the output.")
                return {"answer": answer, "code": code, "output": result["stdout"],
                        "stop_reason": "ok", "attempts": attempts}
            error = result["stderr"] or "Script printed nothing; print the result."
            attempts.append({"code": code, "error": error[-500:]})
    return {"answer": None, "code": None, "output": None, "stop_reason": "max_attempts",
            "attempts": attempts}


if __name__ == "__main__":
    result = code_execution("Compound interest: $500 deposited at the end of every month for "
                            "30 years at 6% annual interest compounded monthly. Final balance?")
    print(result["stop_reason"], "|", result["output"])
    print(result["answer"])
