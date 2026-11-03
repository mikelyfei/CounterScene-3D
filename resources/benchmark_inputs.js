// Recorded participant messages only; evaluator targets are not used here.
export function groupMessages(messages,stageCount){
  const stages=Array.from({length:stageCount},()=>[]),instructions=[];
  let initial=null,stage=0,lastTool='';
  messages.forEach((message,sourceIndex)=>{
    if(message.role==='system'){instructions.push(message);return;}
    if(message.role==='assistant'){
      try{lastTool=JSON.parse(message.content).tool||'';}catch{lastTool='';}
      return;
    }
    if(message.role!=='user')return;
    if(!initial){initial=message;return;}
    let next=false;
    try{
      const value=JSON.parse(message.content);
      if(value.execution_committed===true&&Number.isInteger(value.stage)){
        if(value.stage<0||value.stage>=stageCount)throw new RangeError('Recorded stage is outside the benchmark');
        stage=value.stage;next=true;
      }
    }catch(error){if(error instanceof RangeError)throw error;}
    stages[stage].push({message,sourceIndex,label:next?'New stage observation':`Feedback after ${lastTool||'tool action'}`});
  });
  return {initial,instructions,stages};
}

let indexPromise,version=0;
const traceCache=new Map();
const $=id=>document.getElementById(id);
function setText(id,value){$(id).textContent=value;}
function imageGallery(id,images=[]){
  const host=$(id);host.replaceChildren();
  for(const [i,row] of images.entries()){
    const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption'),link=document.createElement('a');
    image.src=row.file;image.loading='lazy';image.alt=`Actual participant input image ${i+1}`;
    link.href=row.file;link.target='_blank';link.rel='noopener';link.append(image);
    caption.textContent=`Input view ${i+1}`;figure.append(link,caption);host.append(figure);
  }
}
async function getJson(file){const response=await fetch(file);if(!response.ok)throw Error('Recorded participant inputs are unavailable');return response.json();}
function clearTrace(){
  setText('recordedPrompt','');setText('recordedInstruction','');setText('recordedObservation','');
  imageGallery('receivedImages');imageGallery('observationImages');$('observationSelect').replaceChildren();
}
export async function updateBrief(item,stageIndex,execution){
  const current=++version;
  if(!$('participantBrief'))return;
  setText('inputError','');clearTrace();setText('inputInstance',`${item.instance} / Stage ${stageIndex+1}`);
  try{
    indexPromise??=getJson('inputs/index.json');
    const index=await indexPromise;if(current!==version)return;
    const entry=index.instances[item.instance],family=index.families[item.family];
    setText('briefGoal',family.goal);setText('briefOutput',family.output);setText('briefScoring',family.scoring);
    setText('briefCapability',family.capabilities);setText('stageRequirement',family.stages[stageIndex]);
    $('packagePromptLink').href=entry.prompt.file;
    $('packagePromptLink').download=item.instance+'-prompt.txt';
    setText('promptHash',entry.prompt.sha256);
    const response=await fetch(entry.prompt.file);if(!response.ok)throw Error('Task prompt is unavailable');
    const bytes=await response.arrayBuffer();
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==entry.prompt.sha256)throw Error('Task prompt integrity mismatch');
    if(current!==version)return;
    setText('packagePrompt',new TextDecoder().decode(bytes));imageGallery('packageImages',entry.images);
    const selected=entry.runs.find(row=>row.selection===execution);
    $('recordedInputs').hidden=!selected;$('chooseRun').hidden=!!selected;
    if(!selected)return;
    setText('traceCaption',`${selected.model} / repeat ${selected.repeat} / Stage ${stageIndex+1}`);
    $('conversationLink').href=selected.file;
    if(!traceCache.has(selected.file))traceCache.set(selected.file,getJson(selected.file));
    const messages=await traceCache.get(selected.file);if(current!==version)return;
    const groups=groupMessages(messages,item.stages.length);
    setText('recordedInstruction',groups.instructions.map(row=>row.content).join('\n\n'));
    setText('recordedPrompt',groups.initial?.content||'No initial prompt was retained.');
    imageGallery('receivedImages',groups.initial?.images);
    const observations=groups.stages[stageIndex];
    if(!observations.length){setText('recordedObservation','No additional observation was recorded for this stage.');return;}
    for(const [i,row] of observations.entries())$('observationSelect').add(new Option(`${i+1}. ${row.label}`,String(i)));
    function showObservation(){
      const row=observations[Number($('observationSelect').value)];
      setText('recordedObservation',row.message.content);imageGallery('observationImages',row.message.images);
    }
    $('observationSelect').onchange=showObservation;showObservation();
  }catch(error){if(current===version)setText('inputError',error.message);}
}
