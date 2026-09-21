'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="playground-token"]').content;
const config = JSON.parse(document.querySelector('meta[name="playground-config"]').content);
const providers = config.providers;
const groupLabel = providers.length===3?'All three':'Both models';
const titles = {nimble:'Nimble · H100',mac:'Nimble · MacBook',jev:'Jev'};
const subtitles = {nimble:'9B · RUNPOD · BF16',mac:'9B · M5 PRO · MLX BF16',jev:'TYPESAFE · JEV 1.13.0'};
const threeWay = r => r && providers.every(p => r.results[p]);
const names = {choice:'Choice',noul:'Boolean',score:'Score'};
let examples=[], runs=[], selected=null, latest=new Map(), status={}, busy=false, stop=false, batch=false, activeRun=null;
function el(tag, cls, text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function human(s){return String(s).replaceAll('_',' ');}
function label(v){return typeof v==='boolean'?(v?'True':'False'):human(v);}
function ms(v){return Number.isFinite(v)?Math.round(v).toLocaleString()+' ms':'—';}
function median(a){a=a.slice().sort((x,y)=>x-y);return a.length?(a[(a.length-1)>>1]+a[a.length>>1])/2:null;}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Playground-Token':token},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data;}
function notice(text,error=false){$('notice').hidden=!text;$('notice').textContent=text;$('notice').classList.toggle('error',error);}
function sourceText(row){return typeof row.input.state==='string'?row.input.state:JSON.stringify(row.input.state,null,2);}
function currentInput(){const input=structuredClone(selected.input);input.state=typeof selected.input.state==='string'?$('context').value:JSON.parse($('context').value);return input;}
function edited(){return selected&&$('context').value!==sourceText(selected);}
function controls(){ $('run').disabled=busy||!status.ready||!selected;$('run-all').disabled=(!status.ready||busy)&&!batch;$('run-all').textContent=batch?'Stop after this example':'Run all 30';$('export').disabled=runs.length===0;$('reset').disabled=busy;$('context').disabled=busy;}
function renderList(){const nav=$('examples');nav.replaceChildren();const filter=$('kind-filter').value;examples.forEach((row,i)=>{if(filter!=='all'&&filter!==row.kind)return;const button=el('button','sample'+(selected?.id===row.id?' selected':''));button.setAttribute('aria-label','Example '+(i+1)+': '+human(row.domain));button.setAttribute('aria-current',selected?.id===row.id?'true':'false');button.append(el('span','sample-index',String(i+1).padStart(2,'0')));const copy=el('span','sample-copy');copy.append(el('span','sample-domain',human(row.domain)),el('span','sample-sub',names[row.kind]+' · '+row.id));button.append(copy);const r=latest.get(row.id);if(r)button.append(el('span','sample-outcome'+(r.agreement===null?' error':r.agreement?'':' disagree'),r.agreement===null?'!':r.agreement?'✓':'≠'));button.onclick=()=>select(row);nav.append(button);});}
function renderSummary(){
 const records=examples.map(r=>latest.get(r.id)).filter(threeWay);
 const valid=records.filter(r=>r.agreement!==null);
 $('completed').textContent=records.length+' / 30';
 $('failure-count').textContent=records.length?((records.length-valid.length)?(records.length-valid.length)+' comparisons contain errors':groupLabel+' evaluated'):'Ready for a fresh comparison';
 $('agreement').textContent=valid.length?valid.filter(r=>r.agreement).length+' / '+valid.length:'—';
 $('median-times').replaceChildren();
 for(const p of providers){const times=records.map(r=>r.results[p]).filter(a=>a&&!a.error).map(a=>a.request_ms);const line=el('span','median-row');line.append(el('span','',p==='nimble'?'H100':p==='mac'?'MacBook':'Jev'),el('span','',times.length?ms(median(times)):'—'));$('median-times').append(line);}
 controls();
}
function drawCard(provider,record,running=false){const card=$(provider+'-card');card.replaceChildren();const title=el('div','model-title');title.append(el('span','model-mark'),document.createTextNode(titles[provider]));card.append(title,el('p','model-sub',subtitles[provider]));
 if(!record || !record.results[provider]){card.append(el('div','placeholder'+(running?' running':''),running?'Evaluating the same input…':record?'Not run yet · compare again':'Ready to compare'));return;}
 const a=record.results[provider];if(a.error){card.append(el('p','error-message',a.error));return;}
 card.append(el('div','prediction'+(record.kind==='score'?' score':''),record.kind==='score'?a.expected_score.toFixed(2):label(a.prediction)));
 if(record.kind==='score')card.append(el('div','answer-extra','Weighted score · most likely level '+a.prediction));
 if(record.reference_available)card.append(el('span','match'+(a.correct?'':' miss'),a.correct?'Matches reference':'Differs from reference'));
 const timings=el('div','timings');for(const [text,val] of [['RESPONSE TIME',a.request_ms],...(provider!=='jev'?[[provider==='nimble'?'H100 SCORING':'MAC SCORING',a.scoring_ms]]:[])]){const t=el('div','time');t.append(el('strong','',ms(val)),el('span','',text));timings.append(t);}card.append(timings);
 for(const [key,p] of Object.entries(a.probabilities)){const row=el('div','prob-label'+(String(a.prediction)===key?' winner':''));row.append(el('span','',human(key)),el('span','',(p*100).toFixed(1)+'%'));row.firstChild.title=key;const bar=el('progress');bar.max=1;bar.value=p;bar.setAttribute('aria-label',human(key)+': '+(p*100).toFixed(1)+' percent');card.append(row,bar);}}
function renderResult(record,running=false){providers.forEach(p=>drawCard(p,record,running));$('comparison-status').textContent=running?(Object.keys(record?.results||{}).length+' / '+providers.length+' returned · waiting for '+providers.filter(p=>!record?.results[p]).map(p=>p==='nimble'?'H100':p==='mac'?'MacBook':'Jev').join(', ')):record?(!threeWay(record)?'Previous run · some backends not included':record.agreement===null?'One or more requests failed':record.agreement?groupLabel+' agree':'Different most likely answers')+' · '+new Date(record.created_utc).toLocaleTimeString():'Run an example to compare';$('raw').textContent=JSON.stringify(record||selected?.input,null,2);}
function select(row){selected=row;$('sample-number').textContent='EXAMPLE '+String(examples.indexOf(row)+1).padStart(2,'0')+' / 30';$('domain').textContent=human(row.domain);$('type-badge').textContent=names[row.kind];$('context').value=sourceText(row);$('question').textContent=row.input.questions.decision.instructions;const criteria=row.input.questions.decision.criteria;$('criteria').replaceChildren();Object.entries(criteria).forEach(([key,value])=>{const line=el('div','criterion');line.append(el('strong','',human(key)),document.createTextNode(value));$('criteria').append(line);});$('options-summary').textContent=Object.keys(criteria).length+' candidate definitions';$('reference').textContent='Reference: '+label(row.reference)+' · model-checked';renderResult(activeRun?.id===row.id?activeRun:latest.get(row.id),activeRun?.id===row.id);renderList();controls();}
async function streamComparison(body,onEvent){
 const response=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json','Accept':'application/x-ndjson','X-Playground-Token':token},body:JSON.stringify(body)});
 if(!response.ok){const data=await response.json();throw Error(data.error||'Request failed');}
 const reader=response.body.getReader(), decoder=new TextDecoder();let buffer='';
 function consume(line){if(line.trim()){const event=JSON.parse(line);if(event.type==='error')throw Error(event.error);onEvent(event);}}
 try{while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});let newline;while((newline=buffer.indexOf('\n'))!==-1){consume(buffer.slice(0,newline));buffer=buffer.slice(newline+1);}if(done){consume(buffer);break;}}}
 finally{reader.releaseLock();}
}
async function runOne(row,input){
 activeRun={id:row.id,input,kind:row.kind,results:{},agreement:null,reference_available:false};
 renderResult(activeRun,true);let result=null;
 try{
  await streamComparison({id:row.id,input},event=>{
   if(event.type==='start')activeRun=event.record;
   if(event.type==='result')activeRun.results[event.provider]=event.answer;
   if(event.type==='complete'){result=event.record;activeRun=result;}
   if(selected.id===row.id){renderResult(activeRun,!result);$('reference').textContent=activeRun.reference_available?'Reference: '+label(activeRun.reference)+' · model-checked':'Edited input · no reference label';}
  });
  if(!result)throw Error('Connection ended before every result arrived. Received results are still shown.');
  runs.push(result);if(result.reference_available)latest.set(row.id,result);
  renderList();renderSummary();return result;
 }catch(error){
  if(selected.id===row.id){for(const p of providers)if(!activeRun.results[p])activeRun.results[p]={error:'Result not received. Try again.'};renderResult(activeRun);$('comparison-status').textContent='Comparison interrupted · received results retained';}
  throw error;
 }finally{activeRun=null;}
}
$('run').onclick=async()=>{try{const input=currentInput();busy=true;controls();notice('');await runOne(selected,input);}catch(e){notice(e.message,true);}finally{busy=false;controls();}};
$('run-all').onclick=async()=>{if(batch){stop=true;notice('Stopping after the current comparison.');return;}if(busy)return;batch=busy=true;stop=false;controls();try{for(let i=0;i<examples.length;i++){if(stop)break;select(examples[i]);notice('Comparing '+(i+1)+' of 30 — each model receives the same input.');await runOne(examples[i],examples[i].input);}notice(stop?'Stopped. Completed results are saved.':'All 30 comparisons finished. Select any example to inspect the results.');}catch(e){notice(e.message,true);}finally{busy=batch=false;controls();}};
$('reset').onclick=()=>select(selected);
$('context').oninput=()=>{ $('reference').textContent=edited()?'Edited input · reference scoring is disabled':'Reference: '+label(selected.reference)+' · model-checked';renderResult(edited()?undefined:latest.get(selected.id));};
$('kind-filter').onchange=renderList;
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({dataset:'30 held-out examples; stratified random sample, seed 17; edited inputs are separately marked',results:runs},null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='nimble-jev-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
async function refresh(){try{status=await api('/api/status');$('ready-dot').classList.toggle('ready',status.ready);$('connection').textContent=providers.filter(p=>p!=='jev').map(p=>p==='nimble'?(status.gpu_ready?'H100 ready':'H100 offline'):(status.mac_ready?'MacBook ready':'MacBook starting / offline')).join(' · ');if(status.deadline_unix&&providers.includes('nimble')){const min=Math.max(0,Math.ceil((status.deadline_unix*1000-Date.now())/60000));$('remaining').textContent=min?'H100 ends in '+min+' min':'H100 session ended';}controls();}catch(e){$('connection').textContent='Local app disconnected';}}
function configurePage(){
 for(const p of ['nimble','mac','jev'])$(p+'-card').hidden=!providers.includes(p);
 $('results-grid').classList.toggle('two-models',providers.length===2);
 $('subtitle').textContent='One input. '+providers.map(p=>p==='nimble'?'RunPod H100':p==='mac'?'MacBook':'Jev').join(' vs. ')+'.';
 $('agreement-label').textContent=groupLabel+' give the same answer';
 $('remaining').hidden=!providers.includes('nimble');
 $('context-hint').textContent='Editable · sent to '+(providers.length===2?'both models':'all three backends');
 if(!config.hosted && !providers.includes('nimble')){
  $('result-notes').replaceChildren(el('p','','Response times are measured from this MacBook. Nimble runs locally through MLX; Jev uses its HTTPS API. Mac scoring excludes the local request overhead. Each result appears independently. Jev does not expose model-only scoring time.'),el('p','','Nimble uses the 2,826-example adapter with merged BF16 weights and a FP32 candidate head. References are synthetic, model-checked labels and are never sent to either model. Scores show the weighted level; agreement uses the most likely level.'));
  $('footer').replaceChildren(el('span','','Nimble 9B · M5 Pro · MLX BF16'),el('span','','Jev 1.13.0 · TypeSafe API'),el('span','','Credentials stay on this Mac.'));
 }
 if(config.hosted){
  $('result-notes').replaceChildren(el('p','', 'Response times are measured from this RunPod server. H100 runs on the same machine; Jev uses its HTTPS API. Browser-to-RunPod transport is excluded. H100 scoring excludes request overhead. Each result appears as soon as it arrives; Jev does not expose model-only scoring time.'),el('p','', 'Nimble 9B uses the adapter trained on 2,826 examples. References are synthetic, model-checked labels and are never sent to either model. Probabilities compare the supplied candidates. Score agreement uses the most likely level; weighted scores are also shown.'));
  $('footer').replaceChildren(el('span','','Nimble 9B · 2,826 training examples · BF16 · RunPod H100'),el('span','','Jev 1.13.0 · TypeSafe API'),el('span','','Protected session · credentials stay on the server.'));
 }
}
async function init(){try{configurePage();const [data,saved]=await Promise.all([api('/api/examples'),api('/api/results')]);examples=data.examples;runs=saved.results;for(const r of runs)if(r.reference_available)latest.set(r.id,r);select(examples[0]);renderSummary();await refresh();setInterval(refresh,15000);}catch(e){notice(e.message,true);}}
init();
