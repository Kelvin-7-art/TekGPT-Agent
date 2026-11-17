import io,json
from datetime import datetime,timezone
from xml.sax.saxutils import escape
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet


def flatten(value,path=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from flatten(v,f'{path}.{k}' if path else str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from flatten(v,f'{path}[{i}]')
    else:yield [path,value]


def excel_safe(v):
    # Never turn user-controlled strings into spreadsheet formulas.
    return "'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v


def export_report(title,name,overview,history,cleaning,chats,root,kind):
    sections={'Dataset Overview':overview,'Data Quality':{k:v for k,v in overview.items() if k in ['missing_cells','duplicates','fields']},'Cleaning History':cleaning}
    charts=[]
    for h in history:
        r=h['result'];op=h['operation'];tag=f"{h['id']} · revision {h['revision']}"
        if op=='profile':
            sections.setdefault('Descriptive Statistics',{})[tag]=r.get('numeric');sections.setdefault('Correlation Matrix',{})[tag]={'pearson':r.get('pearson'),'spearman':r.get('spearman')};sections['Data Quality'][tag]={'categorical':r.get('categorical'),'outliers':r.get('outliers')}
        elif op=='statistics':sections.setdefault('Statistical Results',{})[tag]=r
        elif op=='ml':sections.setdefault('Machine Learning Results',{})[tag]=r
        elif op=='chart' and h.get('include_report'):
            sections.setdefault('Chart Data',{})[tag]=r.get('data');charts.append(h)
    interpretations=[m for m in chats if m['role']=='assistant']
    if interpretations:sections['Key Findings']={'AI_interpretations':interpretations,'notice':'Model-generated interpretations; verify against the recorded numerical results.'}
    sections['Limitations']={'scope':'Only successfully executed analyses are included. Each result identifies its dataset revision. Earlier results may describe data before cleaning. No causality or generalisation is established by a single analysis.','date_utc':datetime.now(timezone.utc).isoformat(),'recommendation':'Review missingness, sampling design, leakage and test assumptions before decisions.','conclusion':f'This report documents {len(history)} executed analyses; unperformed tests are omitted.'}
    if kind=='excel':
        wb=Workbook();wb.remove(wb.active)
        for section,content in sections.items():
            ws=wb.create_sheet(section[:31]);ws.append(['Metric / evidence path','Recorded value']);ws.freeze_panes='A2';ws.auto_filter.ref='A1:B1'
            for key,value in flatten(content):ws.append([excel_safe(str(key)[:32000]),excel_safe(str(value)[:32000]) if not isinstance(value,(int,float)) else value])
            ws.column_dimensions['A'].width=65;ws.column_dimensions['B'].width=60
            for cell in ws[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='34312C')
            for row in ws.iter_rows(min_row=2):
                for c in row:c.alignment=Alignment(vertical='top',wrap_text=True);c.number_format='0.0000' if isinstance(c.value,float) else 'General'
        out=io.BytesIO();wb.save(out);return out.getvalue()
    styles=getSampleStyleSheet();story=[Paragraph('TekGPT Agent · Analysis Report',styles['Title']),Paragraph(escape(title),styles['Heading1']),Paragraph(escape(name),styles['Normal']),Spacer(1,16)]
    for section,content in sections.items():
        if section=='Chart Data':continue
        story.append(Paragraph(escape(section),styles['Heading1']))
        for key,value in flatten(content):
            text=escape(f'{key}: {value}')
            # Bound long source strings to prevent pathological pagination, mark truncation.
            if len(text)>2000:text=text[:2000]+' [text truncated; see Excel export]'
            story.append(Paragraph(text,styles['BodyText']))
        story.append(Spacer(1,12))
    for h in charts:
        path=root/(h['id']+'.png')
        if path.exists():story.extend([PageBreak(),Paragraph(escape(h['result']['config'].get('title') or 'Visualization'),styles['Heading1']),Paragraph(f"Dataset revision {h['revision']}",styles['Normal']),Image(str(path),width=480,height=278)])
    out=io.BytesIO();SimpleDocTemplate(out,title=title,leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=40).build(story);return out.getvalue()
