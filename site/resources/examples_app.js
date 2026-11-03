import {updateBrief} from './benchmark_inputs.js';

const data=JSON.parse(document.getElementById('dataset').textContent);
const $=id=>document.getElementById(id);
const instanceOrder=(a,b)=>a.localeCompare(b,'en',{numeric:true});
const modelOrder=name=>name.startsWith('gpt-6.1')?0:name.startsWith('claude')?1:name.startsWith('qwen')?2:3;
data.items.sort((a,b)=>instanceOrder(a.instance,b.instance));
for(const row of data.items)row.models.sort((a,b)=>modelOrder(a.model)-modelOrder(b.model)||instanceOrder(a.id||a.model,b.id||b.model));
for(const table of document.querySelectorAll('table')){
  const headers=[...table.querySelectorAll('thead th')].map(node=>node.textContent.trim());
  if(!['Task','Instance'].includes(headers[0]))continue;
  for(const body of table.tBodies){
    const rows=[...body.rows];rows.sort((a,b)=>instanceOrder(a.cells[0].textContent,b.cells[0].textContent)||(headers[1]==='Model'?modelOrder(a.cells[1].textContent)-modelOrder(b.cells[1].textContent):0));
    body.append(...rows);
  }
}


let item,stage,stageIndex=0,timer,ticket=0,blobs={};
$('status').textContent=data.status;
$('coverage').textContent=`${data.items.length} independently replayed current-version benchmark instances.`;
for(const row of data.items)$('instance').add(new Option(`${row.instance} · ${row.stages.length} ordered operations`,row.instance));

async function show(id,scene,version){
  const response=await fetch(scene.file);if(!response.ok)throw Error('Scene unavailable');
  const bytes=await response.arrayBuffer(),digest=await crypto.subtle.digest('SHA-256',bytes);
  const hash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
  if(hash!==scene.sha256)throw Error('Scene integrity mismatch');if(version!==ticket)return;
  if(blobs[id])URL.revokeObjectURL(blobs[id]);
  blobs[id]=URL.createObjectURL(new Blob([bytes],{type:'model/gltf-binary'}));$(id).src=blobs[id];
}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play physical keyframes';}
function reset(){for(const id of ['before','after']){$(id).cameraTarget='auto auto auto';$(id).cameraOrbit='20deg 60deg 110%';$(id).jumpCameraToGoal();}}
function markChanges(){
  for(const id of ['before','after']){
    const viewer=$(id);viewer.querySelectorAll('[slot]').forEach(node=>node.remove());
    for(const row of stage.markers?.[id]||[]){
      const node=document.createElement('span');node.slot='hotspot-'+row.id;node.textContent=objectLabel(row.id);
      node.style.cssText='background:#fff2bd;color:#203543;border:1px solid #9d7a27;padding:2px 5px;border-radius:4px;font-size:11px';
      const [x,y,z]=row.position;node.dataset.position=`${x} ${z} ${-y}`;node.dataset.normal='0 1 0';viewer.append(node);
    }
  }
}
function objectLabel(id){return (data.object_id_aliases?.[id]||id).replaceAll('_',' ');}
function svgNode(name,attributes){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));return node;}
function drawPlots(){
  $('motionPlots').replaceChildren();
  const gold=item.stages[stageIndex].motion,selected=stage.motion;
  if(!gold&&!selected)return;
  const runs=[$('execution').value==='gold'?null:{name:'Gold',motion:gold,color:'#17798a'},{name:$('execution').value==='gold'?'Gold':(item.models.find(row=>(row.id||row.model)===$('execution').value)?.model||'Model'),motion:selected,color:$('execution').value==='gold'?'#17798a':'#c06022'}].filter(row=>row?.motion);
  for(const [a,b,title] of [[0,1,'XY · top view (m)'],[0,2,'XZ · side view (m)']]){
    const samples=runs.flatMap(row=>row.motion.samples),xs=samples.map(row=>row.tip[a]),ys=samples.map(row=>row.tip[b]);
    const minX=Math.min(...xs)-.1,maxX=Math.max(...xs)+.1,minY=Math.min(...ys)-.1,maxY=Math.max(...ys)+.1;
    const point=p=>[35+430*(p[a]-minX)/(maxX-minX),245-200*(p[b]-minY)/(maxY-minY)];
    const box=document.createElement('div'),heading=document.createElement('h3');heading.textContent=title;
    const svg=svgNode('svg',{viewBox:'0 0 500 290',role:'img','aria-label':title+' physical tip trajectories',style:'width:100%;background:#edf3f5;border-radius:6px'});
    for(const row of runs){
      svg.append(svgNode('polyline',{points:row.motion.samples.map(sample=>point(sample.tip).join(',')).join(' '),fill:'none',stroke:row.color,'stroke-width':2}));
      for(const key of row.motion.keyframes)if(key.events.length){const [cx,cy]=point(key.tip);const circle=svgNode('circle',{cx,cy,r:4,fill:row.color});const tooltip=svgNode('title',{});tooltip.textContent=row.name+' '+key.events.map(event=>event.action+' '+event.ball).join(', ');circle.append(tooltip);svg.append(circle);}
    }
    const label=svgNode('text',{x:20,y:278,fill:'#365763','font-size':11});label.textContent=`${runs.map(row=>row.name+(row.color==='#17798a'?' (teal)':' (orange)')).join(' · ')}; X ${minX.toFixed(2)}–${maxX.toFixed(2)}, ${b===1?'Y':'Z'} ${minY.toFixed(2)}–${maxY.toFixed(2)}`;svg.append(label);box.append(heading,svg);$('motionPlots').append(box);
  }
}
async function motion(){
  try{const version=++ticket,index=Number($('frame').value),frame=stage.frames[index];await show('after',frame,version);if(version!==ticket)return;
    $('motion').textContent=`Physical keyframe ${index+1} / ${stage.frames.length}. Playback speed is illustrative. Continuous certificates establish safety; snapshots do not.`;
    $('frameState').textContent=JSON.stringify(stage.motion?.keyframes[index]||{physical_joints_deg:frame.q,executed_ball_positions:frame.balls},null,2);
  }catch(error){$('error').textContent=error.message;stop();}
}
async function selectStage(index){
  stop();stageIndex=index;const base=item.stages[index],run=item.models.find(row=>(row.id||row.model)===$('execution').value),output=run?.stages[index];
  updateBrief(item,index,$('execution').value);
  const version=++ticket;$('error').textContent='';$('motion').textContent='';
  $('title').textContent=`${item.instance} · Stage ${base.number} / ${item.stages.length}`;
  for(const [n,button] of [...$('stages').children].entries())button.setAttribute('aria-pressed',String(n===index));
  if(run&&!output){stage=base;for(const id of ['before','after'])$(id).removeAttribute('src');$('program').hidden=$('play').hidden=$('frame').hidden=$('frameState').hidden=true;$('views').replaceChildren();$('evaluation').textContent='No submitted execution for this stage.';$('motionPlots').replaceChildren();$('operation').textContent=JSON.stringify(base.operation,null,2);$('error').textContent='This model has no executed artifact for the selected stage.';$('summary').textContent=run.eligible?'Stage not executed.':'Infrastructure-ineligible run; do not interpret as a model zero.';return;}
  stage=output?{...base,...output}:base;reset();markChanges();
  const label=run?run.model+' actual execution':'Offline gold';
  $('summary').textContent=`${label}: ${stage.success?'passed':'failed'}. Changed objects: ${stage.changed_objects.map(objectLabel).join(', ')||'see operation and motion state'}. Declared pending collisions: ${base.pending_collisions}. Display names use neutral object aliases; technical programs retain frozen identifiers. The native asset defines appearance.`;
  $('beforeTitle').textContent=item.family==='C'?'Previous reconstruction / initial target':'Before this operation';
  $('afterTitle').textContent=run?run.model+' result':item.family==='C'?'New target and gold reconstruction':'After offline gold execution';
  $('operation').textContent=JSON.stringify({required:base.operation,active_constraints:base.constraints,model_tools:stage.tools||null},null,2);
  $('evaluation').textContent=JSON.stringify(stage.evaluation||{offline_gold_pass:stage.success,pending_collisions:base.pending_collisions},null,2);
  $('program').hidden=!stage.program;$('program').textContent=stage.program||'';
  $('views').replaceChildren();for(const view of base.views||[]){const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption');image.src=view.file;image.alt='Actual target '+view.name;image.style.maxWidth='280px';caption.textContent='Target view: '+view.name;figure.append(image,caption);$('views').append(figure);}
  $('play').hidden=$('frame').hidden=$('frameState').hidden=!stage.frames.length;$('frame').max=Math.max(0,stage.frames.length-1);$('frame').value=0;
  drawPlots();try{await Promise.all([show('before',stage.before,version),show('after',stage.after,version)]);}catch(error){$('error').textContent=error.message;}
}
function performance(){
  $('performance').replaceChildren();
  for(const model of ['gpt-6.1-sol','claude-opus-5-5']){
    const run=item.models.find(row=>row.model===model&&(row.condition||'structured')==='structured'&&(row.view_condition||'active')==='active'&&(row.repeat||1)===1),heading=document.createElement('h3');heading.textContent=model;$('performance').append(heading);
    const text=document.createElement('p');text.textContent=run?(run.eligible?`${run.success?'Strict success':'Strict failure'} · ${run.calls} model calls · ${run.stages.filter(stage=>stage?.success).length}/${item.stages.length} successful stages`:'Infrastructure-ineligible; outcome excluded from model comparison'):'No primary current-protocol result on this instance.';$('performance').append(text);
    if(run){const detail=document.createElement('pre');detail.textContent=JSON.stringify(run.dimensions,null,2);$('performance').append(detail);}
  }
}
function selectItem(){
  item=data.items.find(row=>row.instance===$('instance').value);$('execution').replaceChildren(new Option('Offline gold (not a model)','gold'));
  for(const run of item.models)$('execution').add(new Option(run.label||run.model,run.id||run.model));
  const firstModel=item.models.find(row=>row.model==='gpt-6.1-sol'&&(row.repeat||1)===1)||item.models[0];if(firstModel)$('execution').value=firstModel.id||firstModel.model;
  $('stages').replaceChildren();for(const [index,row] of item.stages.entries()){const button=document.createElement('button');button.textContent=`Stage ${row.number}`;button.onclick=()=>selectStage(index);$('stages').append(button);}
  $('evidence').replaceChildren();const link=document.createElement('a');link.href=item.evidence;link.textContent='Independent benchmark replay certificate';$('evidence').append(link);performance();selectStage(0);
}
function overview(){
  const table=document.createElement('table');table.style.width='100%';
  const header=document.createElement('tr');for(const text of ['Instance','Model','Outcome','Stages','Calls','Inspect']){const th=document.createElement('th');th.textContent=text;th.style.textAlign='left';header.append(th);}table.append(header);
  for(const current of data.items)for(const run of current.models){
    const row=document.createElement('tr');for(const text of [current.instance,run.label||run.model,run.eligible?(run.success?'Passed':'Failed'):'Ineligible',`${run.stages.filter(stage=>stage?.success).length}/${current.stages.length}`,run.calls]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}
    const cell=document.createElement('td'),button=document.createElement('button');button.textContent='Open';button.onclick=()=>{$('instance').value=current.instance;selectItem();$('execution').value=run.id||run.model;selectStage(0);$('title').scrollIntoView({behavior:'smooth',block:'start'});};cell.append(button);row.append(cell);table.append(row);
  }
  $('overview').replaceChildren(table);
}
$('focus').onclick=()=>{for(const id of ['before','after']){const rows=stage.markers?.[id]||[];if(!rows.length)continue;const p=[0,0,0];for(const row of rows)for(let a=0;a<3;a++)p[a]+=row.position[a]/rows.length;$(id).cameraTarget=`${p[0]}m ${p[2]}m ${-p[1]}m`;$(id).cameraOrbit='20deg 60deg 65%';$(id).jumpCameraToGoal();}};
$('instance').onchange=selectItem;$('execution').onchange=()=>selectStage(stageIndex);$('frame').oninput=()=>{stop();motion();};$('reset').onclick=reset;
$('play').onclick=()=>{if(timer){stop();return;}if(Number($('frame').value)>=stage.frames.length-1)$('frame').value=0;$('play').textContent='Pause';timer=setInterval(()=>{if(Number($('frame').value)>=stage.frames.length-1){stop();return;}$('frame').value=Number($('frame').value)+1;motion();},700);};
if(data.items.length){const requestedFamily=new URL(location.href).searchParams.get('family');const preferred=data.items.find(row=>row.family===requestedFamily&&row.stages.length===3&&row.instance.endsWith('s0'))||data.items.find(row=>row.family==='A'&&row.stages.length===3&&row.models.length)||data.items.find(row=>row.family==='A'&&row.models.length)||data.items.find(row=>row.models.length)||data.items[0];$('instance').value=preferred.instance;selectItem();}


for(const button of document.querySelectorAll('[data-family]'))button.onclick=()=>{const chosen=data.items.find(row=>row.family===button.dataset.family&&row.stages.length===3&&row.instance.endsWith('s0'));if(chosen){$('instance').value=chosen.instance;selectItem();$('demo').scrollIntoView({behavior:'smooth'});}};
