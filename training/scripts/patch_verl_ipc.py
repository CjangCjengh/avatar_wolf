#!/usr/bin/env python
# encoding: utf-8
"""
Patch verl 0.8.0 for vllm 0.26 + torch 2.11 compatibility.

Problem: verl's bucketed weight transfer rebuilds CUDA IPC handles via
`rebuild_ipc`, which assumes torch's share_cuda handle args have >= 7
elements. torch 2.11 changed the handle layout, so `list_args[6] = device_id`
raises `IndexError: list assignment index out of range`, breaking
FSDP -> vllm rollout weight syncing.

Fix: force verl's weight transfer onto the shared-memory path
(`use_shm=True`) by making `verl.utils.device.is_support_ipc()` return False.
SHM transfer is semantically identical, just slightly slower — acceptable
for LoRA-scale syncs and base-weight loads on a single node.

Usage:
    python training/scripts/patch_verl_ipc.py [/path/to/site-packages/verl]
"""
import sys
import os

TARGET = "utils/device.py"
ANCHOR = "    # If CUDA is available, it's a GPU device"
INJECT = ("    # NOTE(camwolf): patched to always return False. torch 2.11\n"
          "    # changed the CUDA IPC handle layout, which breaks verl 0.8's\n"
          "    # rebuild_ipc (`IndexError: list assignment index out of range`).\n"
          "    # Forcing the shared-memory weight-transfer path avoids it.\n"
          "    return False\n"
          "    # --- original body below is dead code, kept for reference ---\n")

def main():
    verl_root = sys.argv[1] if len(sys.argv) > 1 else None
    if not verl_root:
        import verl
        verl_root = os.path.dirname(verl.__file__)
    path = os.path.join(verl_root, TARGET)
    src = open(path, encoding="utf-8").read()
    if "NOTE(camwolf)" in src:
        print("already patched:", path)
        return
    if ANCHOR not in src:
        raise SystemExit(f"patch anchor not found in {path}; "
                         "verl version may differ from 0.8.0")
    open(path, "w", encoding="utf-8").write(
        src.replace(ANCHOR, INJECT + ANCHOR, 1))
    print("patched:", path)

if __name__ == "__main__":
    main()
