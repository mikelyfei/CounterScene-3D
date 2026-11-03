"""Load pre-certified native worlds; never search for a new task during a run."""
import json
import hashlib
from pathlib import Path
from .native_env import NativeSceneEnv
from .native_grounding_env import NativeGroundingEnv
from .native_mechanism_env import NativeMechanismEnv
from .progressive import ProgressiveEnv
from .native_operation_env import NativeOperationEnv
from .partial_repair import NativePartialRepairEnv

PROTOCOL='native-asset-v1'
STAGED_PROTOCOL='native-asset-staged-v2'
OPERATIONS_PROTOCOL='native-operations-v3'


def read_stage_manifest(folder):
    """Bind each stage to distinct retained source, never a cycling pair."""
    folder=Path(folder)
    manifest=json.loads((folder/'stages.json').read_text())
    if manifest.get('protocol') not in (STAGED_PROTOCOL,OPERATIONS_PROTOCOL):
        raise ValueError('Unknown staged protocol')
    count={'L1':1,'L2':2,'L3':3,'k06':1,'k14':2,'k34':3}[folder.name.split('-')[1]]
    rows=manifest.get('stages',[])
    if len(rows)!=count:raise ValueError('Missing explicit stage definitions')
    worlds=[];seen=set();seen_gold=set()
    for row in rows:
        path=(folder/row['file']).resolve()
        if path.parent!=folder.resolve():raise ValueError('Stage source outside instance')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=row['sha256'] or digest in seen:raise ValueError('Stage source mismatch or repeated world')
        world=json.loads(raw)
        if not isinstance(world.get('gold'),dict):raise ValueError('Stage has no executable gold witness')
        gold_digest=hashlib.sha256(json.dumps(world['gold'],sort_keys=True).encode()).hexdigest()
        if gold_digest in seen_gold:raise ValueError('Repeated semantic gold target across stages')
        seen_gold.add(gold_digest);seen.add(digest);worlds.append(world)
    return manifest,worlds


def load_staged_episode(folder,artifacts,condition='structured',view_condition='active',capture_cache=None):
    folder=Path(folder);exp=folder.name[0]
    if condition not in ('execution','scalar','structured') or exp!='B' and condition!='structured':
        raise ValueError('Feedback ablation is B-only')
    if view_condition not in ('active','fixed'):raise ValueError('Unknown view condition')
    manifest,private=read_stage_manifest(folder)
    if exp=='D':
        certificate=json.loads((folder/'native_replay_certificate.json').read_text())
        if manifest.get('motion_contract')!='actual-terminal-state-v1' or not certificate.get('all_clear') or certificate.get('manifest_sha256')!=hashlib.sha256((folder/'stages.json').read_bytes()).hexdigest():raise ValueError('D requires a matching native sequential-motion certificate')
        if len(certificate.get('stages',[]))!=len(private):raise ValueError('Missing native motion stages')
        if any(not p['task'].get('stateful_operation') for p in private):raise ValueError('D stage does not preserve actual terminal state')
        cls=NativeMechanismEnv
    else:cls=NativeGroundingEnv if exp=='C' else NativePartialRepairEnv if private[0]['task_internal'].get('repair_stage') is not None else NativeOperationEnv if private[0]['task_internal'].get('operation') else NativeSceneEnv
    worlds=[]
    for n,p in enumerate(private):
        env=cls(p['task'],p,[12.,0.,0.],Path(artifacts)/f'stage{n}') if exp=='D' else cls(p['task_internal'],p,Path(artifacts)/f'stage{n}')
        env.feedback_condition=condition
        if capture_cache:env.capture_cache=Path(capture_cache).resolve()
        if exp=='C':env.view_condition=view_condition
        worlds.append(env)
    episode=ProgressiveEnv(worlds,len(worlds),condition,explicit_stages=True)
    episode.protocol=manifest['protocol']
    return episode


def load_native_episode(folder,artifacts,condition='structured',world_index=0,view_condition='active',capture_cache=None):
    folder=Path(folder);artifacts=Path(artifacts);exp=folder.name[0]
    if condition not in ('execution','scalar','structured') or exp!='B' and condition!='structured':raise ValueError('Feedback ablation is B-only')
    if world_index not in (0,1) or view_condition not in ('active','fixed'):raise ValueError('Unknown native condition')
    if exp=='D':
        private=json.loads((folder/'private.json').read_text())
        worlds=[NativeMechanismEnv(private['task_internal'],private,offsets,artifacts/f'world{n}') for n,offsets in enumerate(([12.,0.,0.],[-12.,0.,0.]))]
        stages=1
    else:
        cls=NativeGroundingEnv if exp=='C' else NativeSceneEnv;worlds=[]
        for n in (0,1):
            private=json.loads((folder/f'world{n}.json').read_text());env=cls(private['task_internal'],private,artifacts/f'world{n}')
            env.feedback_condition=condition
            if exp=='C':env.view_condition=view_condition
            worlds.append(env)
        stages={'L1':1,'L2':2,'L3':3,'k06':1,'k14':2,'k34':3}[folder.name.split('-')[1]]
    if world_index:worlds=worlds[::-1]
    if capture_cache:
        for world in worlds:world.capture_cache=Path(capture_cache).resolve()
    episode=ProgressiveEnv(worlds,stages,condition);episode.protocol=PROTOCOL
    return episode
