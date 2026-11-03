"""Offline retained-action replay; does not call any model provider."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
from counterscene.native_factory import load_staged_episode
p=argparse.ArgumentParser();p.add_argument('--instance',required=True);p.add_argument('--actions',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--condition',choices=['structured','scalar','execution'],default='structured');p.add_argument('--view-condition',choices=['active','fixed'],default='active');a=p.parse_args()
env=load_staged_episode(ROOT/'evaluator'/a.instance,a.artifacts,condition=a.condition,view_condition=a.view_condition)
env.prompt()
for line in a.actions.read_text(encoding='utf-8').splitlines():
 if env.done:break
 try:action=json.loads(line)
 except ValueError:action=None
 env.step(action)
result=env.final()
a.out.write_text(json.dumps(dict(protocol=env.protocol,result=result,trace=env.log),indent=2),encoding='utf-8')
print(json.dumps(dict(success=result['success'],stages=result.get('stage_executions'))))
if not result['success']:sys.exit(1)
