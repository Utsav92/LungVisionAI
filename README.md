# LungVision AI — research-only 3D pulmonary nodule segmentation / detection

**Research use only. Not validated or authorized for clinical diagnosis, screening, treatment decisions, or patient management.**
Original 3D residual U-Net (11.3 M params at 16 base channels), trained from random initialisation (no pretrained weights) on LIDC-IDRI.

## Reproduce (macOS, Apple Silicon, uv installed)
```bash
./setup.sh                                                     # venv (py3.11), pinned deps, editable install, environment_report.json
.venv/bin/python -m lvdata.download_huggingface --inspect      # sizes/disk check, writes data/manifests/scan_manifest.csv
.venv/bin/python -m lvdata.create_splits                       # persistent patient-level 70/15/15 split (refuses to overwrite)
.venv/bin/python scripts/pilot_cases.py 20 8                   # pick pilot scans; pass to: python -m lvdata.download_huggingface --cases <ids>
.venv/bin/python -m lvdata.validate_dataset                    # reports/dataset_audit.{json,csv,html}
.venv/bin/python scripts/prep_available.py                     # resample to 1 mm, build patch indexes
.venv/bin/python scripts/benchmark.py                          # reports/apple_silicon_benchmark.csv, memory_profile.json
.venv/bin/python train.py --config configs/train_mac.yaml [--resume] [--max-minutes N]
.venv/bin/python -m evaluation.evaluate --ckpt experiments/pilot_unet16/best.pt --split test --out reports/eval_test
.venv/bin/streamlit run app/streamlit_app.py                   # localhost only
.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8000
.venv/bin/python -m pytest tests -q
```
Set `HF_TOKEN` for faster Hugging Face downloads (unauthenticated downloads were ~1 scan/min here).

## Deviations from the brief (and why)
- Package `datasets/` is named **`lvdata/`**: it shadowed the Hugging Face `datasets` dependency.
- Pilot scale: the full set is 70 GB (+ ~equal for 1 mm copies); only a pilot subset is downloaded. Training on the rest = same commands with more `--cases`.
- **Model C (malignancy)**: confirmed-malignancy classifier is BLOCKED by `reports/label_provenance.json` (31 unambiguously linkable labelled nodules; no crosswalk between the TCIA diagnosis sheet and mirror nodule ids). Only a radiologist-suspicion model would be permitted; it is not implemented/trained. No `/scan/malignancy-research` endpoint exists.
- "CPM" is not reported: our FROC (centroid-within-radius matching on default-GT components) is not the official LUNA16 script.

## Data
MedOtter/LIDC-IDRI mirror of TCIA LIDC-IDRI + QIICR annotations, CC BY 3.0. Default GT = nodules with >=3 annotations, voxels at >=50 % consensus; 1-2 reader nodules and <3 mm are background (unannotated != nodule-free). LUNA16 is a strict subset of LIDC-IDRI (not independent). Cite Armato et al. 2011 (doi:10.1118/1.3528204), TCIA doi:10.7937/K9/TCIA.2015.LO9QL9SX, Fedorov et al. 2020 (doi:10.1002/mp.14445). Acknowledgement text is in the dataset card.

## Architecture
stem(16)@1 → enc1(32)@1/2 → enc2(64)@1/4 → enc3(128)@1/8 → bottleneck(256)@1/16 → dec3..dec0 (trilinear up + skip concat + residual block) → 1×1×1 conv → logits (sigmoid at inference). GroupNorm + SiLU.
Loss 0.5·Dice + 0.5·BCE. Fixed HU window [-1000, 400] → [0,1], stored in each checkpoint.

## Known limitations
See `reports/` and the Streamlit "Research Limitations" view. Results from the pilot are not representative of a fully trained model.
