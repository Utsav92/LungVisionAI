"""Write environment_report.json: hardware, OS, Python, torch/MPS, disk, memory, swap."""
import json, platform, subprocess, sys, shutil, psutil
import torch

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return f"error: {e}"

hw = sh("system_profiler SPHardwareDataType")
vm, sw = psutil.virtual_memory(), psutil.swap_memory()
du = shutil.disk_usage(".")
rep = {
    "chip": next((l.split(":")[1].strip() for l in hw.splitlines() if "Chip" in l), None),
    "cores": next((l.split(":")[1].strip() for l in hw.splitlines() if "Total Number of Cores" in l), None),
    "gpu_cores": sh("system_profiler SPDisplaysDataType | grep -i 'Total Number of Cores'"),
    "unified_memory_bytes": int(sh("sysctl -n hw.memsize") or 0),
    "macos": sh("sw_vers -productVersion"),
    "python": sys.version, "platform": platform.platform(),
    "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
    "mps_built": torch.backends.mps.is_built(),
    "disk_free_gb": round(du.free / 1e9, 1), "mem_available_gb": round(vm.available / 1e9, 2),
    "swap_used_gb": round(sw.used / 1e9, 2), "swap_total_gb": round(sw.total / 1e9, 2),
}
json.dump(rep, open("environment_report.json", "w"), indent=2)
print(json.dumps(rep, indent=2))
