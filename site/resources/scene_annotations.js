// Presentation-only object markers. Retained scoring evidence is never changed.
export function annotationEvidence(item,base,stage,isModel){
  const op=base.operation||{},task=new Set([op.host,op.object,...(op.payloads||[]),...(op.editable_objects||[]),...(op.active_objects||[])].filter(Boolean));
  if(item.family==='C')for(const id of Object.keys(stage.after?.centres||base.after?.centres||{}))task.add(id);
  const errors=new Map(),notes=[];
  const add=(id,reason)=>{if(typeof id!=='string')return;id=id.split('#')[0];if(!errors.has(id))errors.set(id,[]);if(!errors.get(id).includes(reason))errors.get(id).push(reason);};
  if(!isModel)return {task,errors,notes};
  const ev=stage.evaluation||{},definitions=new Map((base.constraints||[]).map(row=>[row.id,row]));
  for(const row of ev.constraint_checks||[]){
    if(row.satisfied!==false)continue;
    // Soft infeasibility is a separate diagnostic; a false soft check is not automatically a model error.
    if(item.family==='B'&&definitions.get(row.id)?.priority!=='hard')continue;
    for(const id of row.args||[])add(id,`${row.id}: ${row.type||'constraint'} failed`);
  }
  const pending=new Set((op.pending_collision_pairs||[]).map(pair=>[...pair].sort().join('|')));
  for(const row of stage.annotation_assessment?.physical||ev.physical||[]){
    if(pending.has([...(row.objects||[])].sort().join('|')))continue;
    for(const id of row.objects||[])add(id,`Native check: ${row.check||'physical validity'}`);
  }
  for(const [id,distance] of Object.entries(ev.metric_centre_errors_m||{}))if(distance>0.05)add(id,`Centre error ${distance.toFixed(3)} m > 0.050 m`);
  for(const row of ev.operation_checks||[])if(row.ok===false)add(row.object,'Required operation not achieved');
  for(const id of ev.locked_moved||[])add(typeof id==='string'?id:id.id,'Preserved object moved');
  if(ev.hidden_mean_iou<0.8)notes.push('Hidden-view IoU failed; no per-object IoU locality is inferred.');
  if((ev.detect_fn||0)+(ev.detect_fp||0)>0)notes.push('Checker diagnosis is incomplete; this does not imply a geometric error region.');
  if((ev.conflict_fn||0)+(ev.conflict_fp||0)>0)notes.push('Soft-conflict diagnosis is incomplete; no spatial error is inferred.');
  if(ev.soft_feasible_ok<ev.soft_feasible_total)notes.push('A feasible soft preference is incomplete; per-constraint feasibility is not localized here.');
  if(stage.success===false&&!errors.size)notes.push('This execution failed, but retained checks do not identify a precise object-level region.');
  return {task,errors,notes};
}

function nativePoints(scene,bytes){
  const points={...(scene.centres||{})};
  if(!scene.centres&&bytes){
    try{
      const view=new DataView(bytes),length=view.getUint32(12,true);
      const gltf=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,20,length)));
      for(const node of gltf.nodes||[]){
        // The motion exporter emits world-translated root meshes. Do not guess hierarchical transforms.
        if(node.mesh===undefined||!node.translation||node.matrix||node.children)continue;
        const name=node.name||'';let id;
        if(/^ball ball_\d+$/.test(name))id=name.slice(5);
        else if(/^(cabinet_\d+|column_\d+) body$/.test(name))id=name.split(' ')[0];
        else if(['fixed post','pedestal','moving link 1','moving link 2','joint 1','joint 2','gripper bridge'].includes(name))id=name;
        if(id){const [x,y,z]=node.translation;points[id]=[x,-z,y];}
      }
    }catch{/* Unsupported mesh metadata has no fabricated label position. */}
  }
  Object.assign(points,scene.balls||{});
  return points;
}

export function createAnnotations({getContext,label,getNode}){
  let context=null,points={before:{},after:{}},evidence={task:new Set(),errors:new Map(),notes:[]};
  const state={task:true,errors:true,names:false};
  const controls={task:getNode('showTaskRegions'),errors:getNode('showErrorRegions'),names:getNode('showObjectNames')};
  const position=(id,pane)=>points[pane][id.split('#')[0]];
  function focus(ids){
    for(const pane of ['before','after']){
      const rows=ids.map(id=>position(id,pane)).filter(Boolean);if(!rows.length)continue;
      const p=[0,0,0];for(const row of rows)for(let a=0;a<3;a++)p[a]+=row[a]/rows.length;
      const viewer=getNode(pane);viewer.cameraTarget=`${p[0]}m ${p[2]}m ${-p[1]}m`;viewer.cameraOrbit='20deg 60deg 80%';viewer.jumpCameraToGoal();
    }
  }
  function clear(){
    for(const pane of ['before','after'])getNode(pane).querySelectorAll('[data-scene-annotation]').forEach(node=>node.remove());
  }
  function render(){
    clear();
    const on=Object.values(state).some(Boolean);getNode('toggleAnnotations').textContent=on?'Hide all annotations':'Show all annotations';
    for(const [key,node] of Object.entries(controls))node.checked=state[key];
    if(!context)return;
    for(const pane of ['before','after']){
      const viewer=getNode(pane),rows=new Map();
      if(state.names)for(const [id,p] of Object.entries(points[pane]))rows.set(id,{id,p});
      if(state.task)for(const id of evidence.task){const p=position(id,pane);if(p)rows.set(id,{id,p,task:true});}
      if(state.errors&&pane==='after')for(const [id,reasons] of evidence.errors){const p=position(id,pane);if(p)rows.set(id,{...rows.get(id),id,p,error:true,reasons});}
      if(state.task)for(const waypoint of context.base.operation?.waypoints||[]){
        if(waypoint.position)rows.set('waypoint:'+waypoint.id,{id:waypoint.id,p:waypoint.position,task:true,waypoint:true});
      }
      let index=0;
      for(const row of rows.values()){
        const node=document.createElement('button');node.type='button';node.slot='hotspot-annotation-'+index++;
        node.dataset.sceneAnnotation='true';node.className='scene-marker'+(row.task?' scene-marker-task':'')+(row.error?' scene-marker-error':'');
        const [x,y,z]=row.p;node.dataset.position=`${x} ${z} ${-y}`;node.dataset.normal='0 1 0';
        const name=label(row.id),description=row.reasons?.join('; ')||(row.task?'Current task object or waypoint':'Scene object');
        node.title=name+': '+description;node.setAttribute('aria-label',node.title);
        const dot=document.createElement('span');dot.className='scene-marker-symbol';dot.textContent=row.error?'!':row.task?'·':'';node.append(dot);
        if(state.names){const text=document.createElement('span');text.className='scene-marker-name';text.textContent=name;node.append(text);}
        node.onclick=event=>{event.stopPropagation();if(!row.waypoint)focus([row.id]);};viewer.append(node);
      }
    }
    const list=getNode('highlightEvidence');list.replaceChildren();
    for(const [id,reasons] of evidence.errors){
      const line=document.createElement('li');line.textContent=label(id)+': '+reasons.join('; ');list.append(line);
    }
    for(const note of evidence.notes){const line=document.createElement('li');line.textContent=note;list.append(line);}
    getNode('highlightSummary').textContent=evidence.errors.size?`${evidence.errors.size} objects implicated by retained scoring checks.`:context.isModel?'No localized object-level scoring issue is marked.':'Gold reference: task markers only.';
    getNode('focusErrors').disabled=!evidence.errors.size;
  }
  for(const [key,node] of Object.entries(controls))node.onchange=()=>{state[key]=node.checked;render();};
  getNode('toggleAnnotations').onclick=()=>{const next=!Object.values(state).some(Boolean);for(const key of Object.keys(state))state[key]=next;render();};
  getNode('focusErrors').onclick=()=>focus([...evidence.errors.keys()]);
  return {
    clear(){clear();context=null;points={before:{},after:{}};getNode('highlightEvidence').replaceChildren();getNode('highlightSummary').textContent='No executed scene for this selection.';getNode('focusErrors').disabled=true;},
    setStage(){const c=getContext();context={...c,base:c.item.stages[c.stageIndex]};points={before:{...(c.stage.before?.centres||{}),...(c.stage.before?.balls||{})},after:{...(c.stage.after?.centres||{}),...(c.stage.after?.balls||{})}};evidence=annotationEvidence(c.item,context.base,c.stage,c.isModel);render();},
    scene(pane,scene,bytes){if(!context)return;points[pane]=nativePoints(scene,bytes);render();},
    focusTask(){focus([...evidence.task]);},
  };
}
