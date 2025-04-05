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