import re
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

def count_jax_usage(code: str) -> int:
    if not code:
        return 0

    used_libraries = sum(
        1 for lib in JAX_LIBRARIES
        if re.search(rf"\bimport {re.escape(lib)}\b|\bfrom {re.escape(lib)} import", code)
    )
    used_primitives = sum(
        1 for primitive in JAX_PRIMITIVES
        if re.search(rf"\b{re.escape(primitive)}\(", code)
    )
    used_lax_ops = sum(
        1 for op in JAX_LAX_OPERATIONS
        if re.search(rf"\blax\.{re.escape(op)}\(", code)
    )

    return used_libraries + used_primitives + used_lax_ops
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