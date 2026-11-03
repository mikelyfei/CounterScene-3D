# Task C: execution specification

Reconstruct real catalogue assets from calibrated perspective front, side and top RGB views. Later stages substantially relocate target objects and supply new views; revise the reconstruction.

Capabilities: Metric visual grounding, active observation, uncertainty and updating.

The dependency chain is current observation, candidate construction and optional public verification, irreversible execution, then a newly released observation/work order. Later actions must use the retained actual state. Intermediate faulty executions remain visible; a plausible terminal image cannot erase them.

The reference action count below is the executable gold witness, not an enforced agent plan or an empirical difficulty claim. Agents may need more observations and checks within the listed action cap. L1 is the baseline; later operations add state dependencies and preservation obligations.

| Instance | Ordered operations | Gold tool calls | Maximum actions |
|---|---:|---:|---:|
| C-L1-s0 | 1 | 2 | 14 |
| C-L1-s1 | 1 | 2 | 14 |
| C-L2-s0 | 2 | 4 | 28 |
| C-L2-s1 | 2 | 4 | 28 |
| C-L3-s0 | 3 | 6 | 42 |
| C-L3-s1 | 3 | 6 | 42 |

Output: a JSON tool action containing restricted Python source defining `build(public_params)`, returning the complete native-asset placement map or motion payload declared in the prompt. Required diagnosis fields are separate from placement source. The exact task prompt specifies tool arguments, thresholds and output keys.

Only `agent_input/` and public tool replies are participant inputs. `evaluator/` is private to the scorer and contains the ground-truth criteria, stage definitions and gold witness. The included scorer executes the output; the README summarizes its measurements and limits.
