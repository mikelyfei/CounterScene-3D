"""Program execution boundary for the revised agentic benchmark."""
from . import program


def execute_candidate(source, room):
    try:
        payload=program.execute(source,public_params={'room':room})
    except program.ProgramError as error:
        return dict(executed=False,error=str(error)[:300],payload=None)
    if not isinstance(payload,dict):
        return dict(executed=False,error='build must return an object',payload=None)
    return dict(executed=True,error=None,payload=payload)


def save_candidate(task,source,out):
    """Persist actual executed scene and exact geometry renders, never gold images."""
    import hashlib,json
    from pathlib import Path
    from .spatial_suite import core,grounding
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'candidate.py').write_text(source,encoding='utf-8')
    result=execute_candidate(source,task['room'])
    result.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),renders=[])
    if result['executed']:
        report=core.check_scene(task,result['payload'])
        result['geometry']=report
        (out/'placements.json').write_text(json.dumps(result['payload'],indent=2),encoding='utf-8')
        if report['valid_output']:
            cams=grounding.cameras(task['room'],['reference','front_left','back'])
            for name in cams:
                image=grounding.view_png(task,result['payload'],name,cams)
                file=name+'.png';(out/file).write_bytes(image)
                result['renders'].append(dict(file=file,sha256=hashlib.sha256(image).hexdigest(),camera=cams[name]))
    (out/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def save_mechanism_candidate(task,source,offsets,out):
    """Materialize actual commanded motion for existing Blender renderer."""
    import copy,hashlib,json
    from pathlib import Path
    from types import SimpleNamespace
    from .spatial_suite import kinematics
    from scripts.export_spatial_suite import review_geometry
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'candidate.py').write_text(source,encoding='utf-8')
    result=execute_candidate(source,task['room'])
    result.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),renders=[])
    if result['executed']:
        payload=copy.deepcopy(result['payload'])
        try:payload['keyframes']=[[q[i]+offsets[i] for i in range(3)] for q in payload['keyframes']]
        except (KeyError,TypeError,IndexError):
            result.update(executed=False,error='Invalid command keyframes')
        if result['executed']:
            result['geometry']=kinematics.score_d(task,payload)
            (out/'physical_motion.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
            if result['geometry'].get('reason')!='invalid_output':
                render=out/'render_input';render.mkdir()
                try:
                    geo=review_geometry(SimpleNamespace(task=task,priv={'gold':payload},experiment='D'))
                    (render/'review_geometry.json').write_text(json.dumps(geo),encoding='utf-8')
                    (render/'private.json').write_text(json.dumps(dict(task_internal=task)),encoding='utf-8')
                    (render/'public.json').write_text(json.dumps(dict(experiment='D',instance='anonymous-candidate')),encoding='utf-8')
                    result['render_input']='render_input'
                    result['visualization_note']='Renderer compatibility field gold contains the actual candidate, not the benchmark answer. Midpoint/path images do not certify continuous clearance.'
                except (KeyError,TypeError,ValueError,IndexError) as error:
                    result['render_error']=type(error).__name__
    (out/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
