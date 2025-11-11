#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.temporal_reasoning.region_analysis.run_region_temporal_analysis import main

if __name__ == "__main__":
    main()
