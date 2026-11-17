"""Allowlisted numerical operations. No eval, exec, shell, pickle, or user code."""
import io, json, math, re
import numpy as np
import pandas as pd
from scipy import stats


def safe(value):
    return json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=True), parse_constant=lambda _: None)


def columns(df, names, numeric=False):
    if not names or any(c not in df.columns for c in names):
        raise ValueError('Select existing dataset columns.')
    if numeric and any(not pd.api.types.is_numeric_dtype(df[c]) for c in names):
        raise ValueError('This analysis needs numerical columns.')
    return names


def overview(df):
    return safe({'rows':len(df),'columns':len(df.columns),'memory_bytes':int(df.memory_usage(deep=True).sum()),'duplicates':int(df.duplicated().sum()),'missing_cells':int(df.isna().sum().sum()),'fields':[{'name':c,'dtype':str(df[c].dtype),'numeric':bool(pd.api.types.is_numeric_dtype(df[c])),'missing':int(df[c].isna().sum()),'unique':int(df[c].nunique())} for c in df], 'preview':df.head(25).replace({np.nan:None}).to_dict('records')})


def profile(df):
    num=df.select_dtypes(include='number'); numeric=[]; categorical=[]; outliers=[]
    for c in num:
        s=num[c].dropna(); q1,q3=s.quantile([.25,.75]) if len(s) else (np.nan,np.nan); iqr=q3-q1
        count=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()); z=int((np.abs(stats.zscore(s))>3).sum()) if len(s)>2 and s.std()>0 else 0
        numeric.append({'column':c,**s.describe().to_dict(),'median':s.median(),'mode':s.mode().iloc[0] if len(s) else None,'variance':s.var(),'skewness':s.skew(),'kurtosis':s.kurt(),'missing':int(num[c].isna().sum())})
        outliers.append({'column':c,'iqr_count':count,'iqr_percent':100*count/len(s) if len(s) else 0,'zscore_count':z,'nonmissing_rows':len(s)})
    for c in df.select_dtypes(exclude='number'):
        s=df[c]; categorical.append({'column':c,'unique':int(s.nunique()),'high_cardinality':bool(s.nunique()>50),'frequencies':s.value_counts().head(30).to_dict(),'blank_or_whitespace':int(s.dropna().astype(str).str.strip().eq('').sum()),'case_or_whitespace_variants':int(s.nunique()-s.dropna().astype(str).str.strip().str.lower().nunique())})
    return safe({'overview':overview(df),'numeric':numeric,'categorical':categorical,'outliers':outliers,'pearson':num.corr().to_dict(),'spearman':num.corr(method='spearman').to_dict(),'limitations':['Correlation is not causation. Outlier flags are screening rules, not proof of errors. Z-score screening is most useful for roughly normal distributions. Sample standard deviation uses ddof=1. Category frequency tables show up to 30 values.']})


def filtered(df,p):
    d=df.copy()
    if p.get('filter_column'):
        c=columns(d,[p['filter_column']])[0]; op=p.get('filter_operator','equals'); value=p.get('filter_value','')
        if op=='equals': d=d[d[c].astype(str)==str(value)]
        elif op=='contains': d=d[d[c].astype(str).str.contains(str(value),case=False,regex=False,na=False)]
        elif op in ['gt','lt']:
            columns(d,[c],True); v=float(value); d=d[d[c]>v] if op=='gt' else d[d[c]<v]
        else: raise ValueError('Unsupported filter operator.')
    if p.get('date_column'):
        c=columns(d,[p['date_column']])[0]; dates=pd.to_datetime(d[c],errors='coerce',utc=True)
        if dates.isna().any(): raise ValueError('Date column contains missing or invalid dates; clean it first.')
        if p.get('date_start'):d=d[dates>=pd.to_datetime(p['date_start'],utc=True)]
        if p.get('date_end'):d=d[dates.loc[d.index]<pd.to_datetime(p['date_end'],utc=True)+pd.Timedelta(days=1)]
    if d.empty:raise ValueError('No rows remain after filtering.')
    return d


def clean(df,p):
    d=df.copy(); op=p.get('operation'); c=p.get('column')
    if op=='duplicates':return d.drop_duplicates()
    if op=='filter':return filtered(d,p)
    columns(d,[c])
    if op=='drop_column':d=d.drop(columns=[c])
    elif op=='rename':
        name=str(p.get('value','')).strip()
        if not name or name in d.columns or len(name)>100:raise ValueError('Use a unique non-empty column name (up to 100 characters).')
        d=d.rename(columns={c:name})
    elif op=='dtype':
        dtype=p.get('value')
        if dtype=='number':d[c]=pd.to_numeric(d[c],errors='raise')
        elif dtype=='date':d[c]=pd.to_datetime(d[c],errors='raise',utc=True).astype(str)
        elif dtype=='text':d[c]=d[c].astype('string')
        else:raise ValueError('Choose number, date or text.')
    elif op=='trim':d[c]=d[c].astype('string').str.strip()
    elif op=='missing':
        strategy=p.get('strategy','median')
        if strategy=='drop_rows':d=d.dropna(subset=[c])
        elif strategy in ['mean','median']:
            columns(d,[c],True); value=getattr(d[c],strategy)()
            if pd.isna(value):raise ValueError('Cannot impute an entirely missing column.')
            d[c]=d[c].fillna(value)
        elif strategy=='mode':
            mode=d[c].mode()
            if mode.empty:raise ValueError('No observed values to calculate a mode.')
            d[c]=d[c].fillna(mode.iloc[0])
        elif strategy=='constant':d[c]=d[c].fillna(float(p['value']) if pd.api.types.is_numeric_dtype(d[c]) else str(p.get('value','')))
        else:raise ValueError('Unsupported missing-value strategy.')
    else:raise ValueError('Unsupported cleaning operation.')
    if d.empty or len(d.columns)==0:raise ValueError('Cleaning would remove all records or columns.')
    return d


def chart(df,p):
    import plotly.express as px
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=filtered(df,p); kind=p.get('kind','bar'); x=p.get('x'); y=p.get('y'); group=p.get('group') or None; agg=p.get('aggregation','mean'); n=int(p.get('limit',20)); missing=p.get('missing','drop'); color=p.get('color','#bca984')
    if not 1<=n<=100:raise ValueError('Display between 1 and 100 categories.')
    if not 2<=int(p.get('bins',20))<=100:raise ValueError('Histogram bins must be between 2 and 100.')
    if 'chronological' in p and not isinstance(p['chronological'],bool):raise ValueError('Chronological ordering must be true or false.')
    if not re.fullmatch(r'#[0-9a-fA-F]{6}',color):raise ValueError('Use a hex chart colour.')
    if agg not in ['sum','mean','median','count','min','max','std']:raise ValueError('Unsupported aggregation.')
    if group:columns(d,[group])
    if kind=='heatmap':
        num=d.select_dtypes(include='number')
        if num.shape[1]<2:raise ValueError('Heatmaps need at least two numeric columns.')
        if num.shape[1]>30:raise ValueError('Limit the dataset to 30 numeric columns for a readable heatmap.')
        method=p.get('method','pearson')
        if method not in ['pearson','spearman']:raise ValueError('Choose Pearson or Spearman.')
        values=num.corr(method=method); f=px.imshow(values,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1); used=values.reset_index().to_dict('records')
    else:
        columns(d,[x]); relevant=[x]+([y] if y else [])+([group] if group else [])
        if kind in ['histogram','box']:columns(d,[x],True)
        elif kind=='scatter':columns(d,[x,y],True)
        elif agg!='count':columns(d,[y],True)
        if missing=='drop':d=d.dropna(subset=relevant)
        elif missing=='zero':
            for c in relevant:
                if pd.api.types.is_numeric_dtype(d[c]):d[c]=d[c].fillna(0)
                else:d[c]=d[c].fillna('(Missing)')
        else:raise ValueError('Choose drop or zero/category replacement for missing values.')
        if d.empty:raise ValueError('No usable data remains.')
        if kind in ['bar','line','area','pie']:
            keys=list(dict.fromkeys([x]+([group] if group and kind!='pie' else [])))
            d=d.groupby(keys,dropna=False).size().reset_index(name='_value') if agg=='count' else d.groupby(keys,dropna=False)[y].agg(agg).reset_index(name='_value')
            if p.get('chronological'):
                d['_date']=pd.to_datetime(d[x],errors='raise',utc=True);d=d.sort_values('_date');d[x]=d['_date'].astype(str);d=d.drop(columns='_date')
            else:d=d.sort_values('_value' if p.get('sort') in ['ascending','descending'] else x,ascending=p.get('sort')!='descending')
            d=d.head(n); y='_value'
        if kind=='pie':
            if (d[y]<0).any() or d[y].sum()<=0:raise ValueError('Pie values must be non-negative with a positive total.')
            f=px.pie(d,names=x,values=y)
        elif kind=='histogram':f=px.histogram(d,x=x,color=group,nbins=min(100,max(2,int(p.get('bins',20)))),color_discrete_sequence=[color])
        elif kind=='box':f=px.box(d,y=x,x=group,color=group,color_discrete_sequence=[color])
        elif kind=='scatter':
            if len(d)>10000:raise ValueError('Filter scatter plots to 10,000 rows or fewer.')
            f=px.scatter(d,x=x,y=y,color=group,color_discrete_sequence=[color])
        elif kind in ['bar','line','area']:
            fn={'bar':px.bar,'line':px.line,'area':px.area}[kind]; kwargs={'x':x,'y':y,'color':group,'color_discrete_sequence':[color]}
            if kind=='bar' and p.get('orientation')=='horizontal':kwargs.update(x=y,y=x,orientation='h')
            f=fn(d,**kwargs)
        else:raise ValueError('Unsupported chart type.')
        used=d.head(10000).to_dict('records')
    title=str(p.get('title') or kind.title())[:150];f.update_layout(title=title,template='plotly_dark',paper_bgcolor='#202020',plot_bgcolor='#202020',font_color='#e8e5df',margin=dict(l=60,r=30,t=60,b=65))
    if p.get('x_label'):f.update_xaxes(title=str(p['x_label'])[:100])
    if p.get('y_label'):f.update_yaxes(title=str(p['y_label'])[:100])
    # Independent static rendering from the same calculated chart data for PDF.
    fig,ax=plt.subplots(figsize=(10,5.8));ax.set_title(title)
    if kind=='heatmap':
        im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=ax);ax.set_xticks(range(len(values)),values.columns,rotation=60,ha='right');ax.set_yticks(range(len(values)),values.index)
    elif kind=='pie':ax.pie(d[y],labels=d[x].astype(str),autopct='%1.1f%%')
    elif kind=='bar' and group:
        d.pivot(index=x,columns=group,values=y).plot(kind='barh' if p.get('orientation')=='horizontal' else 'bar',ax=ax,rot=45)
    elif kind=='histogram':
        groups=d.groupby(group) if group else [('All',d)]
        for label,g in groups:ax.hist(g[x],bins=int(p.get('bins',20)),alpha=.6,label=str(label))
        if group:ax.legend()
    elif kind=='box':
        groups=list(d.groupby(group)) if group else [(x,d)];ax.boxplot([g[x].dropna() for _,g in groups],tick_labels=[str(k) for k,_ in groups])
    else:
        groups=d.groupby(group) if group else [('All',d)]
        for label,g in groups:
            xx=g[x].to_numpy() if kind=='scatter' else g[x].astype(str).to_numpy(); yy=g[y].to_numpy()
            if kind=='scatter':ax.scatter(xx,yy,label=str(label),alpha=.7)
            elif kind=='bar':(ax.barh if p.get('orientation')=='horizontal' else ax.bar)(xx,yy,label=str(label),alpha=.7)
            elif kind=='area':ax.fill_between(xx,yy,label=str(label),alpha=.5)
            else:ax.plot(xx,yy,label=str(label))
        if group:ax.legend()
        ax.tick_params(axis='x',rotation=45)
    if kind=='bar' and p.get('orientation')=='horizontal' and not p.get('x_label') and not p.get('y_label'):
        ax.set_xlabel(p.get('y') or 'Count');ax.set_ylabel(x)
    elif kind!='heatmap':ax.set_xlabel(p.get('x_label') or x or '');ax.set_ylabel(p.get('y_label') or (p.get('y') if kind not in ['histogram','box'] else ('Frequency' if kind=='histogram' else x)) or '')
    fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);plt.close(fig)
    return {'figure':json.loads(f.to_json()),'data':safe(used),'config':p,'rows_after_filter':len(filtered(df,p)),'limitations':['Category charts display the configured category limit; chart data exports are limited to 10,000 rows.']},buf.getvalue()


def statistics(df,p):
    kind=p.get('test','correlation'); x=p.get('x'); y=p.get('y'); notes=['Observational associations do not establish causation. Independence must be assessed from study design. No adjustment for multiple comparisons.'];out={}
    if kind in ['correlation','chi_square'] and x==y:raise ValueError('Choose two different columns.')
    if kind=='correlation':
        columns(df,[x,y],True);d=df[[x,y]].dropna()
        if len(d)<3 or d.nunique().min()<2:raise ValueError('Need at least three complete pairs and nonconstant variables.')
        a=stats.pearsonr(d[x],d[y]);b=stats.spearmanr(d[x],d[y]);out={'n':len(d),'pearson_r':a.statistic,'pearson_p':a.pvalue,'spearman_r':b.statistic,'spearman_p':b.pvalue}
    elif kind=='confidence_interval':
        columns(df,[x],True);s=df[x].dropna()
        if len(s)<3:raise ValueError('At least three observations required.')
        ci=stats.t.interval(.95,len(s)-1,loc=s.mean(),scale=stats.sem(s));out={'n':len(s),'mean':s.mean(),'95%_lower':ci[0],'95%_upper':ci[1]};notes.append('Student t interval for the population mean; assumes independent observations and approximate normality or adequate sample size.')
    elif kind in ['t_test','anova']:
        columns(df,[x],True);columns(df,[y]);groups=[g[x].dropna().to_numpy() for _,g in df.dropna(subset=[y]).groupby(y)];labels=list(df.dropna(subset=[y]).groupby(y).groups)
        if len(groups)<2 or any(len(g)<3 for g in groups) or len(groups)>20:raise ValueError('Need 2–20 groups with at least three observations each.')
        if kind=='t_test' and len(groups)!=2:raise ValueError('Welch t-test requires exactly two groups.')
        a=stats.ttest_ind(*groups,equal_var=False) if kind=='t_test' else stats.f_oneway(*groups)
        normal=[stats.shapiro(g[:5000]).pvalue if np.std(g)>0 else None for g in groups];lev=stats.levene(*groups)
        out={'statistic':a.statistic,'p_value':a.pvalue,'groups':[{'label':str(k),'n':len(g),'mean':np.mean(g)} for k,g in zip(labels,groups)],'shapiro_p_values':normal,'levene_p':lev.pvalue}
        notes.append('Welch t-test allows unequal variances.' if kind=='t_test' else 'Classical ANOVA assumes equal variances; a low Levene p-value warns that it may be inappropriate.')
        if any(v is not None and v<.05 for v in normal):notes.append('Normality check flagged at least one group; consider a nonparametric test or robust analysis.')
    elif kind=='chi_square':
        columns(df,[x,y]);tab=pd.crosstab(df[x],df[y])
        if min(tab.shape)<2 or max(tab.shape)>30:raise ValueError('Chi-square needs a 2x2 or larger table with at most 30 categories per column.')
        stat,pv,dof,expected=stats.chi2_contingency(tab);out={'statistic':stat,'p_value':pv,'degrees_of_freedom':dof,'expected_below_5_percent':float((expected<5).mean()*100),'table':tab.to_dict()}
        if (expected<5).any():notes.append('Some expected counts are below five: the chi-square approximation may be unreliable; combine defensible categories or use an exact test.')
    else:raise ValueError('Unsupported statistical test.')
    return safe({'test':kind,'columns':[x,y],'results':out,'limitations':notes})


def machine_learning(df,p):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression,LogisticRegression
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
    features=columns(df,p.get('features',[]));target=p.get('target');model=p.get('model','linear'); notes=['Random 25% holdout with seed 42; not a time-series or group-aware validation. Check that predictors contain no future or target-derived information. This is a baseline, not hyperparameter optimization.']
    if len(df)>20000 or len(features)>30:raise ValueError('Filter ML input to at most 20,000 rows and 30 features.')
    clustering=model in ['kmeans','dbscan','hierarchical']
    if target in features:raise ValueError('The target cannot also be a feature.')
    if not clustering:columns(df,[target]);df=df.dropna(subset=[target])
    if len(df)<20:raise ValueError('At least 20 usable rows required.')
    X=df[features].copy();numeric=list(X.select_dtypes(include='number'));cats=[c for c in features if c not in numeric]
    if any(X[c].nunique()>100 for c in cats):raise ValueError('Categorical predictors must have at most 100 levels; remove identifiers.')
    if any(X[c].isna().all() for c in features):raise ValueError('Remove entirely missing features first.')
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
    if clustering:
        if model in ['dbscan','hierarchical'] and len(df)>3000:raise ValueError('DBSCAN and hierarchical clustering are limited to 3,000 rows.')
        k=int(p.get('clusters',3))
        if not 2<=k<=min(20,len(X)-1):raise ValueError('Choose 2–20 clusters, fewer than the row count.')
        eps=float(p.get('eps',.5))
        if not .001<=eps<=100:raise ValueError('DBSCAN eps must be between 0.001 and 100.')
        z=prep.fit_transform(X);est={'kmeans':KMeans(n_clusters=k,random_state=42,n_init=10),'dbscan':DBSCAN(eps=eps,min_samples=5),'hierarchical':AgglomerativeClustering(n_clusters=k)}[model];labels=est.fit_predict(z);unique=np.unique(labels);valid=labels!=-1;score=metrics.silhouette_score(z[valid],labels[valid],sample_size=min(2000,int(valid.sum())),random_state=42) if 1<len(np.unique(labels[valid]))<valid.sum() else None
        return safe({'model':model,'features':features,'metrics':{'silhouette':score},'clusters':pd.Series(labels).value_counts().to_dict(),'projection':{'x':z[:2000,0].tolist(),'y':z[:2000,1].tolist() if z.shape[1]>1 else [0]*min(len(z),2000),'cluster':labels[:2000].tolist()},'limitations':['Exploratory clustering on standardized and encoded features; silhouette excludes DBSCAN noise (-1). Projection shows the first two transformed features, up to 2,000 rows.']})
    regression=model in ['linear','random_forest_regressor','xgboost']
    y=df[target]
    if regression:columns(df,[target],True)
    else:
        y=y.astype(str)
        if not 2<=y.nunique()<=20 or y.value_counts().min()<4:raise ValueError('Classification needs 2–20 classes, each with at least four records.')
    if model=='svm' and len(df)>3000:raise ValueError('SVM is limited to 3,000 rows.')
    estimators={'linear':LinearRegression(),'random_forest_regressor':RandomForestRegressor(n_estimators=80,max_depth=10,random_state=42,n_jobs=1),'logistic':LogisticRegression(max_iter=1000,random_state=42),'decision_tree':DecisionTreeClassifier(max_depth=8,random_state=42),'random_forest_classifier':RandomForestClassifier(n_estimators=80,max_depth=10,random_state=42,n_jobs=1),'svm':SVC(probability=True,random_state=42)}
    if model=='xgboost':
        try:
            from xgboost import XGBRegressor
            estimators['xgboost']=XGBRegressor(n_estimators=80,max_depth=4,random_state=42,n_jobs=1)
        except ImportError:raise ValueError('XGBoost is not installed on this Python service.')
    if model not in estimators:raise ValueError('Unsupported model.')
    a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=None if regression else y);pipe=Pipeline([('preprocess',prep),('model',estimators[model])]);pipe.fit(a,c);pred=pipe.predict(b)
    if regression:met={'MAE':metrics.mean_absolute_error(d,pred),'MSE':metrics.mean_squared_error(d,pred),'RMSE':np.sqrt(metrics.mean_squared_error(d,pred)),'R_squared':metrics.r2_score(d,pred)}
    else:
        met={'accuracy':metrics.accuracy_score(d,pred),'precision_weighted':metrics.precision_score(d,pred,average='weighted',zero_division=0),'recall_weighted':metrics.recall_score(d,pred,average='weighted',zero_division=0),'F1_weighted':metrics.f1_score(d,pred,average='weighted',zero_division=0),'confusion_matrix':metrics.confusion_matrix(d,pred,labels=pipe.classes_).tolist(),'class_labels':pipe.classes_.tolist()}
        try:met['ROC_AUC']=metrics.roc_auc_score(d,pipe.predict_proba(b)[:,1]) if len(pipe.classes_)==2 else metrics.roc_auc_score(d,pipe.predict_proba(b),multi_class='ovr',labels=pipe.classes_)
        except ValueError:met['ROC_AUC']=None
    est=pipe.named_steps['model'];importance=getattr(est,'feature_importances_',None);coefficients=getattr(est,'coef_',None);names=pipe.named_steps['preprocess'].get_feature_names_out().tolist()
    equation=None;diagnostics=None
    if model=='linear':
        residual=np.asarray(d)-pred
        diagnostics={'holdout_residual_mean':float(residual.mean()),'holdout_residual_std':float(residual.std(ddof=1)),'residual_normality_p':float(stats.shapiro(residual).pvalue) if len(residual)>=3 and residual.std()>1e-10 else None}
        if not cats:
            scaler=pipe.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'];raw_coef=np.asarray(est.coef_)/scaler.scale_;raw_intercept=float(est.intercept_-np.dot(raw_coef,scaler.mean_));equation={'raw_intercept':raw_intercept,'raw_coefficients':dict(zip(numeric,raw_coef.tolist()))}
    return safe({'model':model,'target':target,'features':features,'train_rows':len(a),'test_rows':len(b),'metrics':met,'regression_equation':equation,'residual_diagnostics':diagnostics,'coefficients':dict(zip(names,coefficients.tolist())) if model=='linear' else None,'intercept':float(est.intercept_) if model=='linear' else None,'feature_importance':dict(zip(names,importance.tolist())) if importance is not None else None,'predictions':{'actual':d.iloc[:200].tolist(),'predicted':pred[:200].tolist()},'limitations':notes+(['Linear coefficients refer to standardized numeric predictors and encoded categories, not raw-unit coefficients.'] if model=='linear' else [])})
