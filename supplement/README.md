# Learning-oriented validation extension

This extension adds five completed frontier controls, eight matched-budget small-model arms (30 proposals), and two actual-reward refinement studies (four new generations and two reused starters). One planned C continuation is administratively incomplete and excluded, rather than scored as a model failure.

The original best-of-four pools yield mean hidden quality 0.075 under uniform random selection and 0.20 under public-reward selection; gains are concentrated in A. The extension separates execution readiness (26/30 native-valid proposals) from task completion. Twenty-nine proposals receive zero admissible reward, identifying a concrete curriculum and feedback-density opportunity. None of the eight episodes or two direct-reward studies reaches strict completion; this is development evidence about where learning support is needed, not a test of trained RL performance.

The main design contribution is the combination of retained actual state, native execution evidence, targeted shortcut controls and reward/outcome separation. This makes learning-oriented diagnostics inspectable while keeping the primary safety and functional criteria fixed.

Run the repository's default `python reproduce.py --out reproduced` to reconstruct all supplementary rewards, first-tie public selection, random expectations and actual-signal selections from the included native evidence. This is numerical evidence reconstruction; it is distinct from fresh model generation or fresh native rendering. `scaffold.py` preserves the experimental JSON-to-Python adapter; it serializes model numbers, never supplies target answers. Completed control action files can also be passed to the corresponding task package scorer with their original feedback/view condition.

## Areas for improvement

Increase benchmark scale and reserved instances, seed diversity and adaptive optimization horizons; compare curriculum schedules and feedback density before short RL studies. The new arms use24576 context/2048 output tokens, versus 16384/4096 in the original probe, so cross-version differences are not isolated format effects. D uses up to 16 frames (its tested gold fits this cap); the E scaffold evaluates physical repair with constraint-ID diagnosis, while complete physical-pair diagnosis should add the objects field. These experimental interface choices do not change the full frontier task packages.
