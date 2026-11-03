"""Bounded hidden Blender materialisation for actual physical mechanism motion."""
import json,subprocess
from pathlib import Path
from .asset_scene import blender_binary


def render_motion(task,physical,out,preview=False):
    root=Path(__file__).resolve().parents[1];out=Path(out).resolve()
    if out.exists():raise ValueError('Motion artifact exists')
    out.parent.mkdir(parents=True,exist_ok=True);request=out.with_suffix('.request.json');log=out.with_suffix('.blender.log')
    if request.exists() or log.exists():raise ValueError('Motion attempt evidence exists')
    request.write_text(json.dumps(dict(task=task,physical=physical,preview=preview),indent=2))
    with log.open('w') as handle:
        result=subprocess.run([str(blender_binary()),'--background','--factory-startup','--threads','4','--python-exit-code','1','--python',str(root/'scripts/native_mechanism_worker.py'),'--','--request',str(request),'--out',str(out)],stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
    if result.returncode or not (out/'result.json').exists():raise RuntimeError('Native motion export failed; retained log: '+str(log))
    return json.loads((out/'result.json').read_text())
