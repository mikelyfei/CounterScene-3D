# CounterScene-3D

An executable benchmark of stateful spatial decisions using actual native 3D assets. Five task families cover relocation and restoration, progressive collision repair, changing-view reconstruction, sequential manipulation, and repair under misleading feedback.

The comparison evaluates GPT-6.1 Sol and Claude Opus 5.5. Astra supplies auxiliary visual ratings; Qwen3.5:9B (Q4_K_M) supplies a separate best-of-four optimization probe. This is a small development cohort, not a claim of held-out generalization or real-world robotics reliability.

## Inspect the evidence

- Open `site/index.html` through a local HTTP server to inspect stage changes, actual model operations, native geometry and gold/model motion.
- Read `report/report.pdf` and its accompanying LaTeX source.
- `benchmark_results.tsv` contains one row per task/model. Missing dimensions are NA. Across-instance, across-generation and same-output scoring variability are distinct.
- `studies.json` contains reward calibration, attacks, model behavior, auxiliary judgments and independently audited small-model selection.
- `sessions/` contains participant inputs, actual images and executed actions. `agent_session/` separately documents the coding workflow and disclosed privacy redactions. English project prompts are in `agent_session/PROJECT_PROMPTS.json`, with actual submitted patches under `agent_session/selected_code/`.
- Each folder in `tasks/` contains prompts, assets, evaluator targets, a scorer and task-specific failure/hackability analysis.

## Reproduce

Python 3.10 or newer is required. The default command checks release hashes, recalculates rewards from retained native scoring evidence using the included `reward.py`, and regenerates the task/model table:

```sh
python reproduce.py --out reproduced
```

The same command reconstructs visual-judge statistics from criterion-level ratings and checks the best-of-four selections from retained candidate scores. Results are written to `study_reconstruction.json`; this numerical check does not generate new candidates or rerun their native executions.

For native execution, install Blender 4.5 with glTF support and set `COUNTERSCENE_BLENDER` to its executable. This command also replays every participant action sequence and all thirty gold instances:

```sh
python reproduce.py --out reproduced_native --replay
```

On Windows, keep checkout and output paths short because Blender renders inherit platform path limits. The runner uses numbered model directories and records their original identities in `model_replay_index.json`.

Use a new output directory for each reproduction. No provider requests, credentials, model weights or downloads are needed. A correctly reproduced model failure remains a failure; the reproduction check succeeds when its outcome matches the retained result. Retained-evidence reward and table reconstruction is not fresh native scoring; `--replay` additionally checks rewards computed from new native executions.

To inspect a single task, run its `scorer.py` using the command in that package's README. Evaluator targets and gold are for offline verification only and must never be exposed to a participant.

Serve the site locally with:

```sh
python -m http.server 8000
```

Then open `http://localhost:8000/site/`. The site uses packaged viewer code and content-addressed native scene resources.

## Interpretation

Strict success requires all required operations, active constraints and preserved accomplishments. Intermediate errors cannot be erased by showing only a plausible final scene. Static repair evaluates endpoint geometry; only explicitly certified paths support continuous-motion claims. Conservative kinematics does not model forces, friction or dynamic stability.

Native-repeat variance columns cover one GPT output per family with two actual scorer executions; unobserved rows are NA. Numerical rescoring, native replay, generation variability and visual-judge repetition are separate studies.

Visual ratings cannot overturn geometric or functional failures. Public reward can disagree with independently measured quality. Best-of-four selection is limited optimization evidence, not RL training. Consult the report and per-task analyses before using any score as a learning objective.

Neutral object names are mapped in `object_id_aliases.json`; retained programs and scoring evidence preserve frozen execution IDs for exact replay. Installed Qwen model metadata and the actual sampling contract are recorded separately in `studies.json`.

Assets retain their provenance and required licenses in each package. External runtimes and model weights are not redistributed.
