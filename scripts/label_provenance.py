"""Phase 13 gate: auditable label-provenance report. Training of any malignancy model must check reports/label_provenance.json."""
import json, re, warnings, pandas as pd
warnings.filterwarnings("ignore")
R = "data/raw/lidc/"
dx = pd.read_excel(R + "aux/tcia-diagnosis-data-2012-04-20.xls", sheet_name=0)
cols = {c: re.sub(r"\s+", " ", c.split("\n")[0]).strip() + " | " + (c.split("\n")[1] if "\n" in c else "") for c in dx.columns}
dx.columns = ["patient_id", "pat_dx", "pat_method", "primary_site"] + [f"n{i}_{k}" for i in range(1, 6) for k in ("dx", "method")]
nod = pd.read_csv(R + "nodules.csv"); scans = pd.read_csv("data/manifests/scan_manifest.csv")
DX = {0: "unknown", 1: "benign/non-malignant", 2: "malignant primary lung", 3: "malignant metastatic"}
M = {0: "unknown", 1: "2y radiological stability", 2: "biopsy", 3: "surgical resection", 4: "progression/response"}
rep = {"purpose": "decide whether a CONFIRMED-malignancy classifier is scientifically supportable",
       "label_sources": {
           "radiologist_suspicion": "nodules.csv malignancy_score = median of 1-5 subjective ratings per nodule (>=1 reader). NOT pathology.",
           "diagnosis_sheet": "aux/tcia-diagnosis-data-2012-04-20.xls (TCIA, 157 patients): patient-level and per-nodule diagnosis + method."},
       "diagnosis_sheet": {"patients": int(dx.patient_id.nunique()), "patients_in_mirror": int(dx.patient_id.isin(scans.patient_id).sum()),
                           "patient_level_dx": {DX[int(k)]: int(v) for k, v in dx.pat_dx.value_counts().items()},
                           "patient_level_method": {M[int(k)]: int(v) for k, v in dx.pat_method.value_counts().items()},
                           "patient_level_dx_by_method": {f"{DX[int(a)]} via {M[int(b)]}": int(n) for (a, b), n in dx.groupby(["pat_dx", "pat_method"]).size().items()}}}
nl = pd.concat([dx[["patient_id", f"n{i}_dx", f"n{i}_method"]].rename(columns={f"n{i}_dx": "dx", f"n{i}_method": "method"}).assign(nodule_slot=i) for i in range(1, 6)]).dropna(subset=["dx"])
nl["dx"] = pd.to_numeric(nl.dx, errors="coerce"); nl["method"] = pd.to_numeric(nl.method, errors="coerce").fillna(0)
nl = nl.dropna(subset=["dx"]); nl["dx"] = nl.dx.astype(int); nl["method"] = nl.method.astype(int)
known = nl[nl.dx > 0]
rep["diagnosis_sheet"]["nodule_level_labelled_rows"] = int(len(nl)); rep["diagnosis_sheet"]["nodule_level_known_dx_rows"] = int(len(known))
rep["diagnosis_sheet"]["nodule_level_known_by_class"] = {DX[int(k)]: int(v) for k, v in known.dx.value_counts().items()}
rep["diagnosis_sheet"]["metastatic_primary_not_lung"] = int(((dx.pat_dx == 3)).sum())
rep["diagnosis_sheet"]["metastatic_rows_dropped_from_lung_cancer_target"] = "metastatic disease is malignant but is NOT primary lung cancer; kept separate"
known_pats = known.patient_id.nunique(); rep["diagnosis_sheet"]["patients_with_any_known_nodule_dx"] = int(known_pats)
rep["linkage_problem"] = ("The sheet's 'Nodule k' index follows the original LIDC XML nodule numbering. The mirror's nodule_id comes from QIICR TrackingUID "
                          "clustering. No crosswalk is provided, so sheet nodules cannot be assigned to specific mirror nodules/masks without manual verification. "
                          "Only patients with exactly ONE clustered nodule in the mirror could be linked unambiguously.")
single = nod.groupby("patient_id").size(); single = single[single == 1].index
rep["linkable_unambiguously"] = {"patients_with_known_dx_and_exactly_one_mirror_nodule": int(known[known.patient_id.isin(single)].patient_id.nunique()),
                                  "by_class": {DX[int(k)]: int(v) for k, v in known[known.patient_id.isin(single)].drop_duplicates("patient_id").dx.value_counts().items()}}
rep["radiologist_suspicion_task"] = {"nodules_with_score": int(nod.malignancy_score.notna().sum()), "score_distribution": {str(k): int(v) for k, v in nod.malignancy_score.round().value_counts().sort_index().items()},
                                     "caveat": "median of readers' subjective 1-5 ratings; ordinal target; inter-reader disagreement is large"}
n_lin = rep["linkable_unambiguously"]["patients_with_known_dx_and_exactly_one_mirror_nodule"]
rep["decision"] = {"radiologist_suspicion_model": "ALLOWED (research, ordinal/regression, labelled as 'radiologist-rated suspicion')",
                   "confirmed_malignancy_model": "BLOCKED" if n_lin < 100 else "REVIEW",
                   "reason": f"only {n_lin} unambiguously linkable labelled nodules; no nodule crosswalk; too few for patient-level train/val/test with CIs"}
json.dump(rep, open("reports/label_provenance.json", "w"), indent=2); print(json.dumps(rep, indent=2))
