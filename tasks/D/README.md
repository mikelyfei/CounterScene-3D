# D: Stateful sequential manipulation

Protocol: native-operations-v3. Six development instances; L1/L2/L3 contain one/two/three ordered operations. A uses the equivalent k06/k14/k34 identifiers. Difficulty is a design hypothesis, not an assumed monotonic empirical ranking.

Grasp a different ball in each stage, place it on a cabinet, release and move onward. Continue from actual preceding joints and released-object positions.

Capabilities: Controller calibration, inverse kinematics, continuous collision checking and event/state planning.

Provide only agent_input and public tool observations to a participant. evaluator contains hidden targets, constraints and gold actions for offline verification. Never expose it to the participant. Each source must define build(public_params), returning all required fixed-asset transforms or a motion payload; public_params supplies the room only. The restricted interpreter disallows imports, attributes, files and network. The full tool contract and exact tolerances are in each prompt and included scorer.

Decisions are irreversible: observe current evidence, construct a program, verify with public tools, commit the current stage, then process the new work order or observation while preserving prior work. A valid wrong stage is retained and cannot be replaced by only a correct final scene. Old-output, fixture-only and stale-observation controls are tested separately in the benchmark audit.

Static scoring uses native mesh support, contact, intersection and boundaries, plus declared functional requirements. B additionally checks jointly feasible soft constraints and infeasibility claims. C requires hidden-view mean object IoU at least 0.8 and every native bounds centre within 0.05 m. D checks continuous conservative arm/tool/payload enclosures, limits, event semantics and actual state carryover; static frames alone do not certify motion. E grades geometry and diagnosis separately. No force dynamics or stability claim is made. Auxiliary VLM ratings are separate, uncertainty-aware visual evidence.

Offline replay requires Python 3.10+ and Blender 4.5 with glTF support. Set COUNTERSCENE_BLENDER to an absolute executable path. No provider calls, credentials, downloads or model weights are needed. From this package run:

    python scorer.py --instance INSTANCE --actions evaluator/INSTANCE/gold_actions.jsonl --artifacts replay_artifacts/INSTANCE --out replay_INSTANCE.json

Use a fresh artifact directory for a new replay. Retained artifacts are content-checked. Gold replay proves executable feasibility, not model capability. Native exports and renders are produced from the actual included assets, not proxy boxes. This is a development cohort; no held-out generalization claim follows from its results.
