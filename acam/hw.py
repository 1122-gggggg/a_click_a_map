"""Host probes: RAM, swap, GPU, disk, and whether a memory-limited scope is available."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIB = 2**30


@dataclass(frozen=True)
class Memory:
    total: int
    available: int
    swap_total: int
    swap_free: int


def read_meminfo(text: str | None = None) -> Memory:
    text = text if text is not None else Path("/proc/meminfo").read_text()
    kib = {line.split(":")[0]: int(line.split()[1]) for line in text.splitlines() if ":" in line}
    return Memory(
        total=kib["MemTotal"] * 1024,
        available=kib["MemAvailable"] * 1024,
        swap_total=kib.get("SwapTotal", 0) * 1024,
        swap_free=kib.get("SwapFree", 0) * 1024,
    )


def gpu() -> dict[str, object] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,compute_cap", "--format=csv,noheader,nounits"],
            text=True,
            timeout=20,
        ).splitlines()[0]
    except (subprocess.SubprocessError, IndexError, OSError):
        return None
    name, total, used, cap = (part.strip() for part in out.split(","))
    return {"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used), "compute_cap": cap}


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def scope_available() -> bool:
    """True when `systemd-run --user --scope` can start a transient cgroup."""

    if not shutil.which("systemd-run"):
        return False
    try:
        return (
            subprocess.run(
                ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=1G", "true"],
                capture_output=True,
                timeout=20,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        return False
