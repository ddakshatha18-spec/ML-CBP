"""External-dataset evaluation (DFUTissue, ComplexWoundDB) and base-paper comparison.

Run from the repo root with ``PYTHONPATH=src``, e.g.
``PYTHONPATH=src python -m external_eval.zero_shot --help``.

Research prototype. Not clinically validated.

The label-mapping, loader and metric code is torch-free on purpose so it can be
tested anywhere; only ``model_utils``, ``zero_shot``, ``train_dfutissue`` and
``eval_lutseg_native`` need PyTorch.
"""

import os

# Must be set before torch is first imported: lets unsupported ops fall back to CPU on Apple MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

IGNORE = 255
OUR_CLASSES = ["Background", "Epithelial", "Slough", "Granulation", "Necrotic", "Other"]
