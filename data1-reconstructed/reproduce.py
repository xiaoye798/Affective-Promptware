"""重算离线保存响应；不启动推理，不执行实验载荷。"""
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
runpy.run_path(str(Path(__file__).resolve().parent / "analysis/compute_rates.py"), run_name="__main__")
