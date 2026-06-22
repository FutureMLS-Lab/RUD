import torch
import torch.fx
from typing import Callable

FX_KERNEL_REPLACEMENTS: dict[Callable, Callable] = {}


def register_replacement(fx_target: Callable):
    """
    Decorator to register an FX graph replacement for a given op target.
    """
    def decorator(fn: Callable) -> Callable:
        FX_KERNEL_REPLACEMENTS[fx_target] = fn
        return fn
    return decorator


def apply_fx_kernel_replacements(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Walk the FX graph and replace matched nodes with registered replacements.
    """

    replacements_made = set()

    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target in FX_KERNEL_REPLACEMENTS:
            replacement_fn = FX_KERNEL_REPLACEMENTS[node.target]

            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    replacement_fn,
                    args=node.args,
                    kwargs=node.kwargs,
                )
                new_node.meta = node.meta.copy()

            node.replace_all_uses_with(new_node)
            target_str = str(node.target)
            gm.graph.erase_node(node)

            replacements_made.add((target_str, replacement_fn.__name__))

    if replacements_made:
        gm.graph.lint()
        gm.recompile()
        for original, replacement in replacements_made:
            print(f"[FX Kernel Replacement] Replaced {original} -> {replacement}")

    return gm



