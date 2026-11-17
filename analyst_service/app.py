"""Private local analysis service. Start with python -m uvicorn analyst_service.app:app."""
import os,io,json,sqlite3,uuid,time,hashlib,hmac,threading,multiprocessing,logging,csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import Response
from pydantic import BaseModel,ConfigDict,Field
from .analysis import overview,profile,clean,chart,statistics,machine_learning,safe
from .reports import export_report

ROOT=Path(os.environ.get('ANALYST_DATA_DIR','./analyst-data')).resolve();ROOT.mkdir(parents=True,exist_ok=True)
LOG=logging.getLogger('tekgpt.analyst');POOL=ThreadPoolExecutor(max_workers=2);LOCK=threading.Lock()
app=FastAPI(title='TekGPT Controlled Analysis Service',docs_url=None,redoc_url=None)

def db():
    c=sqlite3.connect(ROOT/'state.sqlite',timeout=20);c.row_factory=sqlite3.Row
    c.executescript('''CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,owner TEXT,name TEXT,revision INTEGER,cleaning TEXT,chat TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,owner TEXT,dataset TEXT,revision INTEGER,operation TEXT,params TEXT,status TEXT,result TEXT,error TEXT,created REAL,include_report INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY,owner TEXT,path TEXT,kind TEXT);''');return c

def query(sql,args=(),one=False):
    with db() as c:
        rows=c.execute(sql,args);out=rows.fetchone() if one else rows.fetchall()
    return dict(out) if one and out else ([dict(r) for r in out] if not one else None)

def write(sql,args=()):
    with db() as c:c.execute(sql,args)

def dataset(owner,id):
    d=query('SELECT * FROM datasets WHERE id=? AND owner=?',(id,owner),True)
    if not d:raise ValueError('Dataset not found.')
    return d

def folder(d):return ROOT/d['id']
def frame(d,revision=None):return pd.read_csv(folder(d)/f"{d['revision'] if revision is None else revision}.csv",keep_default_na=True)
def jobs(owner,id):
    rows=query("SELECT * FROM jobs WHERE owner=? AND dataset=? AND status='completed' ORDER BY created",(owner,id))
    return [{**r,'params':json.loads(r['params']),'result':json.loads(r['result'])} for r in rows]

def state(owner,id):
    d=dataset(owner,id);history=jobs(owner,id)
    return {'id':d['id'],'name':d['name'],'revision':d['revision'],'overview':overview(frame(d)),'original_overview':overview(frame(d,0)),'cleaning':json.loads(d['cleaning']),'chat':json.loads(d['chat']),'history':history,'active':query("SELECT id,operation,status,error FROM jobs WHERE owner=? AND dataset=? AND status!='completed' ORDER BY created DESC LIMIT 10",(owner,id))}

ALLOWED={
 'profile':set(),
 'chart':{'kind','x','y','group','aggregation','limit','missing','color','method','title','x_label','y_label','orientation','sort','chronological','bins','filter_column','filter_operator','filter_value','date_column','date_start','date_end'},
 'statistics':{'test','x','y'},
 'ml':{'model','features','target','clusters','eps'},
 'preview_clean':{'operation','column','strategy','value','filter_column','filter_operator','filter_value','date_column','date_start','date_end'},
}

def calculate(root,jid):
    global ROOT
    ROOT=Path(root);j=query('SELECT * FROM jobs WHERE id=?',(jid,),True);d=dataset(j['owner'],j['dataset']);df=frame(d,j['revision']);p=json.loads(j['params']);op=j['operation']
    try:
        if op=='profile':result=profile(df)
        elif op=='statistics':result=statistics(df,p)
        elif op=='ml':result=machine_learning(df,p)
        elif op=='chart':
            result,png=chart(df,p);(folder(d)/(jid+'.png')).write_bytes(png)
        elif op=='preview_clean':
            cleaned=clean(df,p);cleaned.to_csv(folder(d)/(jid+'.csv'),index=False);result={'before':overview(df),'after':overview(cleaned),'operation':p,'approval_required':True}
        else:raise ValueError('Unsupported analysis.')
        encoded=json.dumps(safe(result),allow_nan=False)
        prior=query("SELECT coalesce(sum(length(result)),0) total FROM jobs WHERE dataset=?",(d['id'],),True)['total']
        if len(encoded)>2_000_000 or prior+len(encoded)>6_000_000:raise ValueError('Analysis output limit reached. Delete old results, filter the dataset or reduce chart categories.')
        write("UPDATE jobs SET status='completed',result=? WHERE id=?",(encoded,jid))
    except Exception as e:
        message=str(e)[:800] if isinstance(e,(ValueError,KeyError,TypeError)) else 'Analysis failed. Check input types and service logs.'
        LOG.exception('analysis_failed job=%s operation=%s',jid,op);write("UPDATE jobs SET status='failed',error=? WHERE id=?",(message,jid))

def run_job(jid):
    start=time.time();write("UPDATE jobs SET status='running' WHERE id=?",(jid,));process=multiprocessing.get_context('spawn').Process(target=calculate,args=(str(ROOT),jid));process.start();process.join(120)
    if process.is_alive():process.terminate();process.join();write("UPDATE jobs SET status='failed',error='Analysis stopped after 120 seconds. Use fewer rows or features.' WHERE id=?",(jid,))
    elif query('SELECT status FROM jobs WHERE id=?',(jid,),True)['status']=='running':write("UPDATE jobs SET status='failed',error='Analysis worker exited unexpectedly.' WHERE id=?",(jid,))
    LOG.info('analysis_finished job=%s duration=%.3f',jid,time.time()-start)

class RPC(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action:str=Field(max_length=40)
    id:str=''
    name:str=Field(default='',max_length=150)
    content:str=Field(default='',max_length=5_242_880)
    operation:str=''
    params:dict=Field(default_factory=dict)
    job:str=''
    revision:int=0
    role:str=''
    kind:str='pdf'
    title:str=Field(default='Dataset analysis',max_length=150)
    include:bool=True

async def identity(request):
    key=os.environ.get('ANALYST_SERVICE_TOKEN','')
    if len(key)<32:raise HTTPException(503,'Set a service token of at least 32 characters.')
    if not hmac.compare_digest(request.headers.get('authorization',''),f'Bearer {key}'):raise HTTPException(401,'Unauthorized service request.')
    owner=request.headers.get('x-tekgpt-owner','')
    if not owner or len(owner)>200:raise HTTPException(401,'Missing owner identity.')
    return owner

@app.get('/health')
async def health():return {'ok':True,'service':'TekGPT controlled Python analysis'}

@app.post('/rpc')
async def rpc(request:Request):
    owner=await identity(request)
    raw=await request.body()
    if len(raw)>6_000_000:raise HTTPException(413,'CSV request exceeds 5 MB limit.')
    try:
        b=RPC.model_validate_json(raw)
        with LOCK:
            if b.action=='list':return {'datasets':query('SELECT id,name,revision FROM datasets WHERE owner=?',(owner,))}
            if b.action=='upload':
                if not b.name.lower().endswith('.csv'):raise ValueError('Upload a UTF-8 CSV file.')
                if len(b.content.encode())>5_242_880:raise ValueError('CSV is larger than 5 MB.')
                if len(query('SELECT id FROM datasets WHERE owner=?',(owner,)))>=10:raise ValueError('10 dataset limit; delete an unused dataset first.')
                reader=csv.reader(io.StringIO(b.content));header=next(reader,[])
                if not header or len(header)!=len(set(header)) or any(not c.strip() or len(c)>100 for c in header):raise ValueError('CSV needs unique non-empty column names of at most 100 characters.')
                df=pd.read_csv(io.StringIO(b.content),nrows=100001)
                if df.empty or len(df)>100000 or len(df.columns)>100:raise ValueError('Use 1–100,000 records and at most 100 columns.')
                if any(len(row)!=len(header) for row in reader):raise ValueError('CSV rows have inconsistent column counts.')
                id=str(uuid.uuid4());path=ROOT/id;path.mkdir();df.to_csv(path/'0.csv',index=False);write('INSERT INTO datasets VALUES (?,?,?,?,?,?)',(id,owner,b.name,0,'[]','[]'));return state(owner,id)
            if b.action=='job':
                j=query('SELECT * FROM jobs WHERE id=? AND owner=?',(b.job,owner),True)
                if not j:raise ValueError('Job not found.')
                if j['status'] in ['queued','running'] and time.time()-j['created']>300:write("UPDATE jobs SET status='failed',error='Job interrupted; please retry.' WHERE id=?",(b.job,));j.update(status='failed',error='Job interrupted; please retry.')
                return {**j,'result':json.loads(j['result']) if j['result'] else None,'params':json.loads(j['params'])}
            d=dataset(owner,b.id)
            if b.action=='state':return state(owner,b.id)
            if b.action=='delete':
                import shutil
                write('DELETE FROM jobs WHERE dataset=? AND owner=?',(b.id,owner));write('DELETE FROM datasets WHERE id=? AND owner=?',(b.id,owner));shutil.rmtree(folder(d),ignore_errors=True);return {'ok':True}
            if b.action=='start':
                if b.operation not in ALLOWED or set(b.params)-ALLOWED[b.operation]:raise ValueError('Unsupported operation or parameters.')
                if len(json.dumps(b.params))>16000:raise ValueError('Analysis parameters too large.')
                active=query("SELECT id FROM jobs WHERE owner=? AND status IN ('queued','running')",(owner,))
                if len(active)>=2:raise ValueError('Two analyses already running. Wait for completion.')
                if len(jobs(owner,b.id))>=50:raise ValueError('50-analysis limit for this dataset. Delete old analyses.')
                jid=str(uuid.uuid4());write('INSERT INTO jobs(id,owner,dataset,revision,operation,params,status,created) VALUES (?,?,?,?,?,?,?,?)',(jid,owner,b.id,d['revision'],b.operation,json.dumps(b.params),'queued',time.time()));POOL.submit(run_job,jid);return {'job':jid}
            if b.action=='apply_clean':
                j=query("SELECT * FROM jobs WHERE id=? AND owner=? AND dataset=? AND operation='preview_clean' AND status='completed'",(b.job,owner,b.id),True)
                if not j or j['revision']!=d['revision']:raise ValueError('Cleaning preview is stale or unavailable. Create a fresh preview.')
                import shutil
                rev=d['revision']+1;shutil.copyfile(folder(d)/(b.job+'.csv'),folder(d)/f'{rev}.csv');hist=json.loads(d['cleaning']);hist.append({'revision':rev,'operation':json.loads(j['params']),'before_after':json.loads(j['result']),'time':time.time()});write('UPDATE datasets SET revision=?,cleaning=? WHERE id=?',(rev,json.dumps(hist),b.id));return state(owner,b.id)
            if b.action in ['chart_report','delete_result']:
                if not query('SELECT id FROM jobs WHERE id=? AND owner=? AND dataset=?',(b.job,owner,b.id),True):raise ValueError('Result not found.')
                if b.action=='chart_report':write('UPDATE jobs SET include_report=? WHERE id=?',(int(b.include),b.job))
                else:write('DELETE FROM jobs WHERE id=?',(b.job,))
                return {'ok':True}
            if b.action=='chat':
                if b.role not in ['user','assistant']:raise ValueError('Invalid role.')
                if len(b.content)>16000:raise ValueError('Chat message too long.')
                messages=json.loads(d['chat']);messages.append({'role':b.role,'content':b.content,'revision':d['revision'],'time':time.time()});write('UPDATE datasets SET chat=? WHERE id=?',(json.dumps(messages[-50:]),b.id));return {'ok':True}
            if b.action=='clear_chat':write("UPDATE datasets SET chat='[]' WHERE id=?",(b.id,));return {'ok':True}
            if b.action=='export':
                if b.kind not in ['pdf','excel']:raise ValueError('Choose PDF or Excel.')
                content=export_report(b.title,d['name'],overview(frame(d)),jobs(owner,b.id),json.loads(d['cleaning']),json.loads(d['chat']),folder(d),b.kind);eid=str(uuid.uuid4());path=folder(d)/(eid+('.pdf' if b.kind=='pdf' else '.xlsx'));path.write_bytes(content);write('INSERT INTO exports VALUES (?,?,?,?)',(eid,owner,str(path),b.kind));return {'download':eid,'kind':b.kind}
            raise ValueError('Unknown action.')
    except (ValueError,KeyError,TypeError) as e:raise HTTPException(400,str(e)[:1000])

@app.get('/download/{id}')
async def download(id:str,request:Request):
    owner=await identity(request);r=query('SELECT * FROM exports WHERE id=? AND owner=?',(id,owner),True)
    if not r or not Path(r['path']).is_file():raise HTTPException(404,'Report not found.')
    kind=r['kind'];return Response(Path(r['path']).read_bytes(),media_type='application/pdf' if kind=='pdf' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',headers={'Content-Disposition':f'attachment; filename="TekGPT-analysis.{"pdf" if kind=="pdf" else "xlsx"}"'})
