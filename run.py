import sys
import os

# Add code/business_entity_resolution/src to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "code", "business_entity_resolution", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline import main

if __name__ == "__main__":
    main()
