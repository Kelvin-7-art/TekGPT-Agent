import io,os,json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from analyst_service.analysis import profile,clean,chart,statistics,machine_learning
from analyst_service.reports import export_report
from analyst_service import app as service

@pytest.fixture
def df():
    return pd.DataFrame({'x':np.arange(1,81,dtype=float),'y':np.arange(1,81,dtype=float)*5+2,'group':['A','B']*40,'date':pd.date_range('2026-01-01',periods=80).astype(str),'label':['low']*40+['high']*40})

def test_profile_clean(df):
    d=df.copy();d.loc[0,'x']=np.nan;d=pd.concat([d,d.iloc[[1]]],ignore_index=True)
    p=profile(d);assert p['overview']['duplicates']==1;assert p['overview']['missing_cells']==1
    assert abs(p['pearson']['x']['y']-1)<1e-10
    c=clean(d,{'operation':'missing','column':'x','strategy':'median'});assert c.x.isna().sum()==0;assert d.x.isna().sum()==1
    assert len(clean(d,{'operation':'duplicates'}))==80
    with pytest.raises(ValueError):clean(df,{'operation':'rename','column':'x','value':'y'})
    with pytest.raises(ValueError):clean(df,{'operation':'missing','column':'group','strategy':'mean'})

@pytest.mark.parametrize('kind',['bar','line','scatter','histogram','box','heatmap','pie','area'])
def test_all_charts(df,kind):
    result,png=chart(df,{'kind':kind,'x':'x' if kind in ['histogram','box','scatter'] else 'group','y':'y','aggregation':'sum','title':'Actual dataset'})
    assert result['figure']['data'];assert png[:8]==b'\x89PNG\r\n\x1a\n'
    if kind in ['bar','line','pie','area']:assert abs(sum(r['_value'] for r in result['data'])-df.y.sum())<1e-8

def test_chart_filters(df):
    result,_=chart(df,{'kind':'bar','x':'group','y':'y','aggregation':'sum','filter_column':'group','filter_value':'A','date_column':'date','date_start':'2026-01-01','date_end':'2026-01-10'})
    assert result['rows_after_filter']==5
    assert result['data'][0]['_value']==135
    with pytest.raises(ValueError):chart(df,{'kind':'histogram','x':'group'})

def test_statistics(df):
    c=statistics(df,{'test':'correlation','x':'x','y':'y'});assert c['results']['pearson_r']==pytest.approx(1)
    ci=statistics(df,{'test':'confidence_interval','x':'x'});assert ci['results']['95%_lower']<40.5<ci['results']['95%_upper']
    for t in ['t_test','anova']:assert 'p_value' in statistics(df,{'test':t,'x':'y','y':'group'})['results']
    assert 'p_value' in statistics(df,{'test':'chi_square','x':'label','y':'group'})['results']

def test_ml(df):
    r=machine_learning(df,{'model':'linear','features':['x'],'target':'y'});assert r['metrics']['R_squared']==pytest.approx(1);assert r['metrics']['RMSE']<1e-8
    for model in ['logistic','decision_tree','random_forest_classifier','svm']:
        r=machine_learning(df,{'model':model,'features':['x'],'target':'label'});assert 0<=r['metrics']['accuracy']<=1;assert r['train_rows']==60
    for model in ['kmeans','dbscan','hierarchical']:
        r=machine_learning(df,{'model':model,'features':['x','y'],'clusters':2});assert sum(r['clusters'].values())==80
    with pytest.raises(ValueError):machine_learning(df,{'model':'linear','features':['x','y'],'target':'y'})

def test_reports(df,tmp_path):
    from openpyxl import load_workbook
    from pypdf import PdfReader
    profile_result=profile(df);figure,png=chart(df,{'kind':'bar','x':'group','y':'y','aggregation':'sum','title':'Revenue by group'})
    (tmp_path/'chart1.png').write_bytes(png)
    hist=[{'id':'eda1','revision':0,'operation':'profile','result':profile_result},{'id':'chart1','revision':0,'operation':'chart','result':figure,'include_report':True}]
    overview={'rows':80,'columns':5,'missing_cells':0,'duplicates':0,'fields':[]}
    pdf=export_report('Test report','source.csv',overview,hist,[],[],tmp_path,'pdf');reader=PdfReader(io.BytesIO(pdf));assert 'TekGPT' in reader.pages[0].extract_text();assert any(len(p.images)>0 for p in reader.pages)
    xlsx=export_report('Test report','source.csv',overview,hist,[],[{'role':'assistant','content':'=HYPERLINK("bad")'}],tmp_path,'excel');wb=load_workbook(io.BytesIO(xlsx));assert 'Chart Data' in wb.sheetnames;assert 'Machine Learning Results' not in wb.sheetnames
    assert all(c.data_type!='f' for ws in wb for row in ws for c in row)

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('ANALYST_SERVICE_TOKEN','test-only-token-with-at-least-32-characters');monkeypatch.setattr(service,'ROOT',tmp_path)
    # Execute the same job function synchronously for deterministic API integration tests.
    monkeypatch.setattr(service.POOL,'submit',lambda fn,jid:service.calculate(str(tmp_path),jid))
    return TestClient(service.app)

def rpc(client,body,owner='alice'):
    return client.post('/rpc',json=body,headers={'Authorization':'Bearer test-only-token-with-at-least-32-characters','X-TekGPT-Owner':owner})

def test_api_upload_analysis_clean_export_isolation(client,df):
    assert client.post('/rpc',json={'action':'list'}).status_code==401
    r=rpc(client,{'action':'upload','name':'data.csv','content':df.to_csv(index=False)});assert r.status_code==200,r.text;id=r.json()['id'];assert r.json()['overview']['rows']==80
    assert rpc(client,{'action':'state','id':id},'bob').status_code==400
    r=rpc(client,{'action':'start','id':id,'operation':'chart','params':{'kind':'bar','x':'group','y':'y','aggregation':'sum'}});jid=r.json()['job'];j=rpc(client,{'action':'job','job':jid});assert j.json()['status']=='completed',j.text
    assert rpc(client,{'action':'job','job':jid},'bob').status_code==400
    r=rpc(client,{'action':'start','id':id,'operation':'preview_clean','params':{'operation':'rename','column':'x','value':'advertising'}});preview=r.json()['job']
    assert rpc(client,{'action':'state','id':id}).json()['revision']==0
    applied=rpc(client,{'action':'apply_clean','id':id,'job':preview});assert applied.json()['revision']==1
    assert rpc(client,{'action':'apply_clean','id':id,'job':preview}).status_code==400
    assert 'x' in pd.read_csv(service.ROOT/id/'0.csv').columns
    for kind in ['pdf','excel']:
        r=rpc(client,{'action':'export','id':id,'kind':kind});assert r.status_code==200,r.text;eid=r.json()['download'];headers={'Authorization':'Bearer test-only-token-with-at-least-32-characters','X-TekGPT-Owner':'alice'}
        assert client.get('/download/'+eid,headers=headers).status_code==200
        headers['X-TekGPT-Owner']='bob';assert client.get('/download/'+eid,headers=headers).status_code==404
    assert rpc(client,{'action':'start','id':id,'operation':'exec','params':{'code':'print(1)'}}).status_code==400
    assert rpc(client,{'action':'start','id':id,'operation':'profile','params':{'code':'bad'}}).status_code==400

def test_invalid_csv(client):
    for content in ['', 'x,x\n1,2','x,y\n1,2,3']:
        assert rpc(client,{'action':'upload','name':'bad.csv','content':content}).status_code==400

def test_real_process_worker(tmp_path,monkeypatch,df):
    import uuid,time
    monkeypatch.setattr(service,'ROOT',tmp_path);id=str(uuid.uuid4());(tmp_path/id).mkdir();df.to_csv(tmp_path/id/'0.csv',index=False)
    service.write('INSERT INTO datasets VALUES (?,?,?,?,?,?)',(id,'alice','process.csv',0,'[]','[]'))
    jid=str(uuid.uuid4());service.write('INSERT INTO jobs(id,owner,dataset,revision,operation,params,status,created) VALUES (?,?,?,?,?,?,?,?)',(jid,'alice',id,0,'profile','{}','queued',time.time()))
    service.run_job(jid)
    job=service.query('SELECT * FROM jobs WHERE id=?',(jid,),True);assert job['status']=='completed';assert json.loads(job['result'])['overview']['rows']==80
