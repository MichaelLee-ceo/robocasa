const fs=require('fs');
const code=fs.readFileSync((process.env.RC_DIR || '/tmp/claude-1001/rc') + '/page.js','utf8');
const els={};
function mk(id){return els[id]||(els[id]={id,innerHTML:'',_l:{},setAttribute(){},addEventListener(e,f){this._l[e]=f},querySelectorAll(){return []},insertAdjacentHTML(){}, get textContent(){return ''}});}
global.window={}; global.localStorage={getItem(){return null},setItem(){}};
global.document={getElementById:mk,querySelector:(q)=>mk(q),querySelectorAll:()=>[]};
try{ require('vm').runInThisContext(code + `
;globalThis.__DATA=DATA; globalThis.__renderTask=renderTask;`);
  // render every task panel
  for(const t of Object.keys(__DATA.tasks)){ __renderTask(t); if(!els.panel.innerHTML.includes('每一個 atomic 步驟')) throw new Error('panel missing for '+t); }
  console.log('RENDER_OK', Object.keys(__DATA.tasks).length, 'tasks; panel chars', els.panel.innerHTML.length, '; stepvids in last panel', (els.panel.innerHTML.match(/class="mini"/g)||[]).length);
}catch(e){ console.log('ERROR', e.stack.split('\n').slice(0,3).join(' | ')); }
