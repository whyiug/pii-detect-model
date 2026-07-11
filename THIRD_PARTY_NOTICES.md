# Third-Party Notices and Provenance Status

This file describes dependencies and planned sources; it is not a legal conclusion. A release must
generate a source-specific NOTICE/SBOM and pass the license gate using the exact resolved versions,
dataset revisions, teacher revisions, and terms snapshots actually used.

## Runtime and development packages

The phase-A package declares PyYAML as an optional core dependency and declares optional training,
Presidio, and development dependency groups in `pyproject.toml`. Those packages are not vendored.
Their licenses and transitive dependencies must be collected from the resolved environment before
release. In particular:

- PyYAML is distributed under the MIT license.
- Presidio is distributed under the MIT license.
- Transformers, Accelerate, PEFT, Safetensors, Datasets, and Evaluate are generally distributed
  under Apache-2.0, but the exact installed artifacts and transitive packages remain authoritative.
- PyTorch and scientific Python packages have their own license and notice requirements.

## Models and datasets referenced by the plan

No model weights or dataset rows are currently redistributed by this repository.

- Qwen base/teacher checkpoints must be pinned to exact revisions and their model-card licenses
  recorded before use. A base-model license does not by itself determine the license of a trained
  release.
- `ai4privacy/pii-masking-openpii-1.5m` is only a proposed warm-up source. Its exact revision,
  attribution, data provenance, and permission to train publicly distributed derived weights must
  be reviewed before admission to the public pool.
- `wan9yu/pii-bench-zh` is proposed strictly as an evaluation-only source. It must not enter
  training, prompt development, distillation, calibration, or threshold selection.
- Sources with non-commercial, no-derivatives, research-only, unknown, or incompatible terms must
  not enter a public/commercial training path.
- Customer/internal gold data is private by default and is not authorized for public model
  training merely because it can be accessed internally.

## Teacher and API output

Self-hosted and API teachers require independent provenance records. Provider output ownership does
not automatically establish permission to train or publicly distribute derived weights. External
API output is excluded from the public release pool unless a documented review explicitly approves
the exact service, contract, use, and distribution path.

## Required release evidence

Every release must include, at minimum, exact source revisions, license/terms snapshots and hashes,
attribution, sampling evidence, `data_provenance.json`, `teacher_provenance.json`, an SBOM, and
checksums. Unresolved sources remain quarantined.

