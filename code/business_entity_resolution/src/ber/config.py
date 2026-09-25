import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", r"D:\Amazon project\DATA\student_resource\dataset"))
ARTIFACT_DIR = Path(os.environ.get("BER_ARTIFACT_DIR", "artifacts"))
SEED = 42
S2_PREFIX = "S2-"
S3_PREFIX = "S3-"
