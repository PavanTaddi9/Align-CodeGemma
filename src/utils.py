import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeVar,List
from openai.types import Completion


import openai
import tenacity
import tiktoken

N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2


def read_json(path: str | Path) -> Any:
    """Read a JSON file from a path and return the parsed content."""
    with Path(path).open("r") as f:
        return json.load(f)

def write_json(path: str | Path, data: Any, mode: str = "w") -> None:
    """Write the given data as JSON to the specified file."""
    with Path(path).open(mode) as f:
        json.dump(data, f, indent=2)


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def retry(errors: Any, max_attempts: int = 5):
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(errors),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=print,
    )


ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncClient()

    @retry(ERRORS)
    def chat_completions_with_backoff(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    def completions_with_backoff(self, *args, **kwargs):
        return self.client.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def chat_completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.completions.create(*args, **kwargs)

    async def delayed_request(
        self,
        request: dict[str, Any],
        mode: Literal["chat", "completion"],
        delay: float | None,
    ):
        """Prevent quantized rate limit:
        https://help.openai.com/en/articles/6891753-rate-limit-advice"""
        if delay is not None:
            # synchronized sleep
            time.sleep(delay)
        if mode == "chat":
            func = self.chat_completions_with_backoff_async
        else:
            func = self.completions_with_backoff_async
        return await func(**request)

    async def dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch chat completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [self.delayed_request(request, "chat", delay) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def dispatch_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [
            self.delayed_request(request, "completion", delay) for request in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    # encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_fingerprint(*args: Any, hash_length: int | None = None) -> str:
    combined = "".join(map(str, args))
    content = hashlib.sha256(combined.encode()).hexdigest()
    if hash_length is not None:
        content = content[:hash_length]
    return content


def find_code_blocks(response: str, tag: str | None = None) -> list[str]:
    """Find all enclosed code blocks in the response, optionally filtering by language tag."""
    all_indices = find_codeblock_indices(response, tag)
    return [response[start:end].strip() for start, end in all_indices]


def find_codeblock_indices(
    response: str, tag: str | None = None
) -> list[tuple[int, int]]:
    """Find all enclosed code blocks in the response, optionally filtering by language tag."""
    all_indices: list[tuple[int, int]] = []
    search_start = (
        0  # Variable to keep track of where to start searching for the next code block
    )

    while "```" in response[search_start:]:
        # Find the start of the code block (excluding the backticks)
        code_start_index = response.find("```", search_start) + 3

        # Find the end of the language tag line (or the start of the code if no tag line)
        code_start_endline = response.find("\n", code_start_index)
        if code_start_endline == -1:  # Handle case where there's no newline after ```
            code_start_endline = code_start_index

        # Extract the language tag (if any)
        extracted_tag = response[code_start_index:code_start_endline].strip()

        # Adjust the start index if a language tag is found
        if extracted_tag:
            actual_code_start = code_start_endline + 1
        else:
            actual_code_start = code_start_index

        # Find the end of the code block
        code_end_index = response.find("```", actual_code_start)
        if code_end_index == -1:
            break  # Exit if there's no closing ```

        # Extract the code
        # code = response[actual_code_start:code_end_index].strip()

        # Check if the extracted code block matches the requested language tag (if any)
        if tag is None or extracted_tag.lower() == tag.lower():
            all_indices.append((actual_code_start, code_end_index))

        # Update the search_start to look for the next code block
        search_start = code_end_index + 3

    return all_indices


DEFAULT_TEMPLATE = """\
### Instruction
{instruction}

### Response
{response}"""


JAX_LIBRARIES = [
    "jax", "jax.numpy", "jax.random", "jax.scipy", "jax.experimental",
    "jax.lax", "jax.ops", "jax.tree_util", "flax", "optax", "equinox", "orbax"
]

JAX_PRIMITIVES = [
    "jit", "grad", "vmap", "pmap", "scan", "checkpoint", "remat",
    "custom_jvp", "custom_vjp", "value_and_grad"
]

JAX_LAX_OPERATIONS = [
    "cond", "scan", "map", "while_loop", "dynamic_slice", "dynamic_update_slice",
    "gather", "scatter", "reduce_window", "sort", "stop_gradient"
]

def flatten_openai_responses(responses: List[Completion]) -> List[str]:
    """Flatten OpenAI responses into a list of code strings."""
    return [choice.text for response in responses for choice in response.choices]

def count_jax_usage(code: str) -> int:
    """
    Counts the number of JAX-related imports, primitives, and lax operations used in the code.
    """
    used_libraries = sum(1 for lib in JAX_LIBRARIES if re.search(rf"(^|\s|;)import {lib}|from {lib} import", code))
    used_primitives = sum(1 for primitive in JAX_PRIMITIVES if re.search(rf"(^|\s|;){primitive}\(", code))
    used_lax_ops = sum(1 for op in JAX_LAX_OPERATIONS if re.search(rf"lax\.{op}\(", code))
    
    return used_libraries + used_primitives + used_lax_ops