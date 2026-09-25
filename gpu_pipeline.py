"""
Root runner for GPU Entity Resolution Pipeline
"""
import sys
import os

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from gpu_pipeline import run_gpu_entity_resolution

if __name__ == "__main__":
    run_gpu_entity_resolution()
