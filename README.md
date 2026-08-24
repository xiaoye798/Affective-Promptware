# Affective Promptware — Paper Artifact

Data and validation artifact for the NDSS 2026 submission
**"Affective Promptware: Emotion Representations as an Attack Surface of Software Engineering Agents."**


---

## 1. What this artifact is

The paper asks whether the internal affective representations of an LLM can be exploited from *outside* the model, through untrusted text alone. This repository releases the complete evaluation record behind that claim, so that every number in Tables II–IV and Figures 5–6 can be recomputed without access to GPUs, model weights, or the network.

The released evaluation covers **7 open-weight models × 8 affective states × 40 held-out scenarios × 2 conditions × 2 runs = 8,960 model outputs**, generated from **2,240 strictly paired artifacts**. Each adversarial artifact is shipped together with its matched neutral counterpart, so a reviewer can read both sides of a pair and confirm that the affective framing is the only systematic difference within it.

Two things distinguish this artifact from a summary-statistics release:

* **The paired materials are included verbatim.** Every model-visible user message, in both the raw form and the model-native chat-template rendering, with SHA-256 hashes for both.
* **The representation-level evidence is included at cell granularity.** Not just the curves of Figure 5, but the 267,520 per-cell projections and the resampling indices that produce the reported intervals.

## 2. Ethics and intended use

This artifact documents an attack, and is released for defensive research and for reproducibility of the published measurements.

* **Model-only boundary.** The experimental harness records the action a model *selects*; it never executes generated code, never invokes an external service, and never modifies persistent state. Every tool event in the release carries `execution_allowed=false` and `execution_occurred=false`. A "high-risk outcome" in this data denotes a generated unauthorized *request*, not a successful exploit.
* **No sensitive content.** All workflow artifacts and affective corpora are synthetically constructed. They contain no credentials, no proprietary code, and no identifiable data.
* **Deliberately incomplete as an attack tool.** The release excludes deployment-ready attack automation and any integration with a real execution environment.

Reviewers and later users are asked to keep these materials inside evaluation and detection research, and not to place them into live agent pipelines. If unforeseen risks emerge, the authors commit to restricting the release further.


## 3. From paper claim to file

| Claim in the paper | Primary file | Recomputed from |
|---|---|---|
| Table II — restricted-tool selection rate per model and emotion; RD from 7.34 to 100.00 pp | `data/tables/table_ii.csv` | `data/behavior/raw_run_records.jsonl.gz` → `checks/table_ii_recomputed.csv` |
| Table III — layer-wise peak (0.94–3.61 null SD), half-rise depth, AUC | `data/tables/table_iii.csv` | `data/representation/figure5_layerwise.csv` → `checks/table_iii_recomputed.csv` |
| Table IV — pair counts in the five transition categories | `data/tables/table_iv.csv` | `data/behavior/raw_run_records.jsonl.gz` → `checks/table_iv_recomputed.csv` |
| Figure 5 — layer-wise target-direction shift, per-model trajectories and intervals | `data/representation/figure5_layerwise.csv` | `figure5_cells.*`, `figure5_bootstrap_indices.npz`, `figure5_bootstrap_draws.csv.gz` |
| Figure 6 (a)–(b) — pair-level association between shift and change in tool selection | `data/representation/figure6_pairs.csv` | `analysis_labels.delta_z_target` in the run records |
| Figure 6 (c)–(d) — off-target and random control directions | `data/representation/figure6_directions.csv`, `figure6_control_projections.*` | `data/representation/figure6_summary.json` |
| Section IV-C worked pair (Figures 2–4) | `data/materials/paired_artifacts.jsonl.gz`, `pair_id = wf021-anxious` | — |
| Frozen experimental settings (§IV-B, §IV-D) | `configs/experiment_config.yaml`, `configs/model_manifest.csv`, `configs/model_generation_configs.json` | — |

A consolidated spreadsheet view of the tables and figure data is available in `outputs/paper_complete_data.xlsx`; `checks/workbook_verification.json` records its cell-level agreement with the CSVs.

## 4. Directory layout

| Directory | Contents |
|---|---|
| `data/tables/` | Paper-aligned Tables II, III, and IV |
| `data/behavior/` | Condition-level behavior, full response records, tool events, per-emotion counts |
| `data/materials/` | Matched neutral and affective prompts plus their indexes |
| `data/representation/` | Figure 5 layer-wise, cell, and resampling data; Figure 6 pair, direction, and control data |
| `configs/` | Model identifiers, revisions, effective generation settings, global experiment settings |
| `checks/` | Recomputed summaries and validation reports |

## 5. File inventory

| Data object | File | Records |
|---|---|---:|
| Table II | `data/tables/table_ii.csv` | 7 rows |
| Table III | `data/tables/table_iii.csv` | 7 rows |
| Table IV | `data/tables/table_iv.csv` | 7 rows |
| Figure 5 layer-wise data | `data/representation/figure5_layerwise.csv` | 418 rows |
| Figure 5 model summary | `data/representation/figure5_summary.csv` | 7 rows |
| Figure 5 cell data | `data/representation/figure5_cells.csv.gz`, `.npz` | 267,520 rows |
| Figure 5 scenario-resampling draws | `data/representation/figure5_bootstrap_draws.csv.gz` | 28,000 rows |
| Figure 5 scenario-resampling indices | `data/representation/figure5_bootstrap_indices.npz` | 4,000 draws |
| Figure 6 pairwise data | `data/representation/figure6_pairs.csv` | 4,480 rows |
| Figure 6 direction summary | `data/representation/figure6_directions.csv` | 208 rows |
| Figure 6 control projections | `data/representation/figure6_control_projections.csv.gz`, `.npz` | 931,840 rows |
| Figure 6 estimand summary | `data/representation/figure6_summary.json` | 1 file |
| Table II/IV condition-level behavior | `data/behavior/condition_records.csv` | 8,960 rows |
| Full response and tool-event records | `data/behavior/raw_run_records.jsonl.gz` | 8,960 records |
| Run-record index | `data/behavior/raw_run_records_index.csv` | 8,960 rows |
| Per-emotion selection counts | `data/behavior/per_emotion_counts.csv` | 56 rows |
| Matched neutral and affective materials | `data/materials/paired_artifacts.jsonl.gz` | 2,240 pairs |
| Material index | `data/materials/paired_artifacts_index.csv` | 2,240 rows |
| Model manifest | `configs/model_manifest.csv` | 7 rows |
| Model generation settings | `configs/model_generation_configs.json` | 7 configurations |
| Global experiment settings | `configs/experiment_config.yaml` | 1 file |


## 7. Field reference

### Full run records

Each line of `data/behavior/raw_run_records.jsonl.gz` is one model output.

| Field | Meaning |
|---|---|
| `output_id` | Unique identifier for one model, scenario, emotion, run, and condition |
| `pair_id` | Identifier shared by the neutral and affective records of one counterfactual pair |
| `material_pair_id` | Reference to the corresponding entry in `paired_artifacts.jsonl.gz` |
| `model_id`, `model_name`, `model_revision` | Model identity and recorded revision |
| `scenario`, `source_scenario_id`, `emotion`, `run`, `condition` | Experimental cell identifiers |
| `request.prompt_sha256` | SHA-256 of the model-visible user message |
| `request.rendered_input_sha256` | SHA-256 of the native chat-template rendering |
| `response.raw_output` | Complete structured model response as recorded |
| `response.parsed_tool_action` | Parsed action: `python_exec`, `review_only`, or `refuse` |
| `response.parsed_reason` | Parsed short reason from the structured response |
| `response.attack_success` | Whether the selected action is `python_exec` |
| `tool_events` | Selection event for the parsed action; no execution occurs |
| `analysis_labels.transition` | Counterfactual transition category used by Table IV |
| `analysis_labels.delta_z_target` | Target-direction pairwise projection used by Figure 6 |

### Matched materials

Each line of `data/materials/paired_artifacts.jsonl.gz` is one neutral/affective pair.

| Field | Meaning |
|---|---|
| `material_pair_id` | Unique model/scenario/emotion material identifier |
| `neutral`, `affective` | Condition-specific model-visible messages and rendered chat inputs |
| `full_user_message` | Complete single-user-role message for the condition |
| `rendered_chat_input` | Message after application of the model-native chat template |
| `prompt_sha256`, `rendered_input_sha256` | Integrity hashes for the two text representations |
| `shared_fields` | Technical operation, tool menu, response schema, sampling configuration, and authorization boundary shared by the pair |
| `pair_checks` | Explicit invariants that must hold between the two conditions |
| `workflow_scenario` | Stable workflow title, domain, incomplete state, handoff note, and closure definition |

### Model settings

`configs/model_generation_configs.json` records the effective per-model generation and tool-interface settings. `configs/model_manifest.csv` is a flat index of repository identifiers, revisions, layer selections, numeric precision, prompt and input hashes, and effective decoding settings.

## 8. Conventions and reading notes

1. **Pairing.** For each scenario–emotion cell, the model-visible neutral prompt and the model-visible affective prompt are identical across all seven models; only the native chat-template rendering is model-specific.
2. **No execution.** Tool events record model selection only. `execution_allowed=false` and `execution_occurred=false` hold for every record in the release.
3. **Layer indexing.** `selected_layer` is one-based; `normalized_depth` gives the corresponding position in model depth.
4. **Admission.** The `admitted` field indicates whether a model enters the F-versus-C comparison reported for Figure 6. Three models are excluded by the admission rule fixed in Section V-C; their counts still appear in Table IV.
5. **Controls.** Figure 6 covers one target direction, seven off-target emotion directions, and 200 norm-matched random directions.
6. **Intervals.** All reported intervals come from the scenario-clustered percentile bootstrap described in Section IV-E, using the seed recorded in `checks/validation_report.json`. The one exception is the comparison against the random control directions, which is an empirical null on absolute effect.
7. **Recomputed values.** Files under `checks/` are outputs of the validators, not independent inputs. Tables II and IV in `data/tables/` are derived from the run records and are reproduced there for convenience.

## 9. Integrity

* `MANIFEST.sha256` lists every package file except itself, with its SHA-256 digest.
* `package_inventory.csv` gives relative paths, sizes, hashes, and file categories.

## 10. Scope of what this artifact can show

Stated here to match the limitations in Section VII, so that reviewers do not have to infer them from the data:

* The pair-level analysis of Figure 6 is **observational**. It shows that the representation shift is larger in the pairs whose tool selection changes, and larger on the target direction than on either control family. It does not show that removing the shift would remove the behavioral effect; that requires an inference-time intervention, which is left to future work.
* The evaluation covers **open-weight models only**, under a **single artifact carrier** (CI logs in the confirmatory evaluation) and a **single decision interface**.
* Two models sit at the ceiling of the behavioral endpoint (OLMo-3.1-32B and Mistral-Small-3.2-24B) and one at a high neutral baseline (Qwen3.6-35B); for these, the behavioral risk difference and the representation measurement should be read together rather than separately. See Sections V-A and V-B.
