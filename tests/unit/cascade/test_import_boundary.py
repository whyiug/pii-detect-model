from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_rules_only_cascade_does_not_import_optional_model_stacks() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from pii_zh.cascade import CascadeConfig, CascadePipeline; "
                "assert CascadePipeline(config=CascadeConfig(mode='rules-only')); "
                "assert not {'torch', 'transformers', 'presidio_analyzer'} & sys.modules.keys()"
            ),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
