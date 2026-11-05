"""Minimal, source-bound revision of the existing OCI figure designs.
Run with the bundled Python: python code/render_figures.py [Fig3 ...].
No selector, model fitting, threshold optimization or raw-image editing occurs here.
"""
from pathlib import Path
import sys, json, argparse, io, string, os
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'_runtime'))
sys.path.insert(0,str(OUT/'code'))
os.environ['MPLCONFIGDIR']=str(OUT/'qa/matplotlib_cache')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Wedge,Rectangle,PathPatch,Patch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap,Normalize
from scipy.stats import gaussian_kde
from PIL import Image
try:
    import fitz
except ImportError:
    fitz=None  # Numeric rendering can proceed while the PDF repair dependency installs.
from audit_panel_alignment import require_matplotlib_panel_alignment

D=OUT/'source_data/derived';I=OUT/'source_data/input'
MODEL='ENet_one_se_thr70'
BLUE='#5289b4';RED='#cd6151';MID='#d3d4d5';ORANGE='#e6a16d';DARK='#333333'
COLORS=[BLUE,MID,RED];GRADES=['Compact','Medium','Loose']
CORE=['Struct_num_gaps','Struct_gap_ratio','Color_S']
SHORT={'Struct_num_gaps':'Gap count','Struct_gap_ratio':'Gap ratio','Color_S':'Color_S'}
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':6.5,'axes.titlesize':8,'axes.titleweight':'normal',
 'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'savefig.facecolor':'white','figure.facecolor':'white','legend.frameon':False})
# Use the actual installed Times New Roman files, never silent substitutions.
FONT_FAMILY='Times New Roman'
FONT_REGULAR=font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY),fallback_to_default=False)
FONT_BOLD=font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY,weight='bold'),fallback_to_default=False)
mpl.rcParams.update({'font.family':'serif','font.serif':[FONT_FAMILY],
    'mathtext.fontset':'custom','mathtext.rm':FONT_FAMILY,
    'mathtext.it':FONT_FAMILY+':italic','mathtext.bf':FONT_FAMILY+':bold',
    'mathtext.fallback':None})

def read(n): return pd.read_csv(D/n)
def jread(n):return json.loads((I/n).read_text(encoding='utf-8'))
def write(n,v): (OUT/'qa'/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def figdir(name):
    p=OUT/('supplementary' if name.startswith('Supplement') else 'figures')/name
    p.mkdir(parents=True,exist_ok=True);return p
def newfig(h):return plt.figure(figsize=(190/25.4,h/25.4))
def axat(fig,box,key):
    ax=fig.add_axes(box,label=key)
    return ax
def letter(ax,s,offset=(-27,8)):
    ax.annotate('('+s+')',(0,1),xycoords='axes fraction',xytext=offset,
                textcoords='offset points',ha='left',va='bottom',weight='bold',fontsize=9,annotation_clip=False)
def save(fig,name,axes,rows=None,columns=None,exemptions=(),dpi=1000):
    p=figdir(name);fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=[a.get_label() for a in axes],
       row_groups=rows,column_groups=columns,exemptions=exemptions,
       json_out=OUT/'qa'/f'{name}.alignment.json',overlay_svg=OUT/'qa'/f'{name}.alignment.svg',
       tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    # Use the exact final page dimensions; no tight-bbox rescaling.
    for ext in ['pdf','svg']:
        fig.savefig(p/f'{name}.{ext}',format=ext,dpi=600)
    fig.savefig(p/f'{name}.tiff',dpi=dpi,pil_kwargs={'compression':'tiff_lzw'})
    preview=fitz.open(p/f'{name}.pdf')
    preview[0].get_pixmap(dpi=300,alpha=False).save(p/f'{name}.png');preview.close()
    layout={a.get_label():dict(xlim=list(map(float,a.get_xlim())),ylim=list(map(float,a.get_ylim()))) for a in axes}
    write(f'{name}.axis_limits.json',dict(width_mm=190,height_mm=float(fig.get_figheight()*25.4),tiff_dpi=dpi,axes=layout))
    plt.close(fig)

def box(ax,data,positions,colors,width=.55):
    b=ax.boxplot(data,positions=positions,widths=width,patch_artist=True,whis=1.5,
       showfliers=True,medianprops={'color':'white','linewidth':.8},
       whiskerprops={'color':'#555555','linewidth':.65},capprops={'color':'#555555','linewidth':.65},
       flierprops={'marker':'.','markersize':1.6,'markerfacecolor':'#444444','markeredgecolor':'none'})
    for patch,c in zip(b['boxes'],colors):patch.set(facecolor=c,alpha=.8,edgecolor='#555555',linewidth=.65)
    return b

def Fig2():
    # Same full-matrix design and variable order as the supplied correlation script.
    corr=pd.read_csv(D/'Fig2_correlation_matrix_275.csv',index_col=0)
    n=len(corr);fig=newfig(185);ax=axat(fig,[.225,.245,.665,.71],'a')
    cmap=LinearSegmentedColormap.from_list('OriginalBlueWhiteRed',['#3976bd','#f7f7f7','#b1363b'])
    for i in range(n):
        for k in range(n):
            r=float(corr.iloc[i,k]);x,y=k+.5,i+.5
            if i>k:
                # White gutters retain the original grid appearance without
                # white stroke paths through merged PDF text extraction spans.
                ax.add_patch(Rectangle((k+.01,i+.01),.98,.98,facecolor=cmap((r+1)/2),edgecolor='none'))
                ax.text(x,y,f'{r:.2f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(r)>.55 else 'black')
            elif i<k:
                ax.add_patch(Circle((x,y),.40,facecolor='white',edgecolor='#cccccc',lw=.35))
                a0,a1=(90-abs(r)*360,90) if r>=0 else (90,90+abs(r)*360)
                ax.add_patch(Wedge((x,y),.40,a0,a1,facecolor=cmap((r+1)/2),edgecolor='none'))
    for k in [0,1]:
        ax.add_patch(Rectangle((0,k),n,1,fc='#aaaaaa',alpha=.06,ec='none'))
        ax.add_patch(Rectangle((k,0),1,n,fc='#aaaaaa',alpha=.06,ec='none'))
    ax.add_patch(Rectangle((0,0),2,2,fill=False,ec='black',lw=.9))
    ax.set(xlim=(0,n),ylim=(n,0));ax.set_aspect('equal',adjustable='box')
    ax.set_xticks(np.arange(n)+.5,corr.columns,rotation=90,ha='right',rotation_mode='anchor',fontsize=6.5)
    ax.set_yticks(np.arange(n)+.5,corr.index,fontsize=6.5)
    ax.tick_params(length=0);[sp.set_visible(False) for sp in ax.spines.values()]
    cbax=fig.add_axes([.915,.35,.013,.50]);fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-1,1),cmap=cmap),cax=cbax)
    cbax.set_ylabel('Pearson r',fontsize=7);cbax.tick_params(labelsize=6.5)
    save(fig,'Fig2',[ax],dpi=1000)

def Fig3():
    pv=read('Fig3_explained_variance.csv');ld=read('Fig3_variable_PC_correlations.csv').set_index('feature')
    scores=read('Fig3_scores_275_training_fit_projection.csv')
    fig=newfig(247)
    a=axat(fig,[.27,.78,.60,.16],'a');b=axat(fig,[.27,.46,.60,.235],'b');c=axat(fig,[.27,.075,.60,.30],'c')
    x=np.arange(1,9)
    a.bar(x,pv.variance_ratio,color='#69a2bf',edgecolor='black',linewidth=.3,width=.80,label='Individual')
    a.plot(x,pv.cumulative_ratio,'o-',color=RED,lw=1,ms=2.7,label='Cumulative')
    a.axhline(.85,c='#999999',ls='--',lw=.6)
    a.set(xlim=(.4,8.6),ylim=(0,1.04),ylabel='Variance ratio',xlabel='Principal component')
    a.set_xticks(x,[f'PC{k}' for k in x]);a.legend(loc='lower right',bbox_to_anchor=(1,1.02),ncol=2)
    a.text(.68,.78,'85% retention level',transform=a.transAxes,color='#777777',fontsize=7,va='top')
    order=ld.iloc[:,:3].pow(2).sum(axis=1).sort_values(ascending=False).index
    z=ld.loc[order,['PC1','PC2','PC3']].to_numpy()
    im=b.imshow(z,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto',interpolation='nearest')
    b.set_xticks([0,1,2],['PC1','PC2','PC3']);b.set_yticks(range(8),order,fontsize=7)
    b.set(xlabel='Principal component');b.tick_params(length=0)
    for i in range(8):
        for k in range(3):b.text(k,i,f'{z[i,k]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(z[i,k])>.55 else 'black')
    cbax=fig.add_axes([.89,.48,.014,.195]);cb=fig.colorbar(im,cax=cbax);cb.set_label('Variable–PC correlation',fontsize=7)
    for split,marker,label in [('train','o','Training (n = 220)'),('test','^','Historical test projection (n = 55)')]:
        p=scores[scores.Set.eq(split)]
        c.scatter(p.PC1,p.PC2,c=p.OCI_Score_recomputed,cmap='RdBu_r',vmin=0,vmax=100,
                  s=9 if split=='train' else 14,marker=marker,alpha=.80,edgecolors='white',linewidths=.25,label=label)
    c.axhline(0,c='#cccccc',ls=':',lw=.5);c.axvline(0,c='#cccccc',ls=':',lw=.5)
    # Preserve the original biplot style. Arrow endpoints represent correlations
    # multiplied by a documented display factor, not PCA score units.
    scale=4.1
    label_pos={'Color_B':(-4.9,2.3),'Color_S':(4.6,2.9),'Size_Perimeter':(3.4,-4.35),
       'Struct_num_gaps':(6.0,-3.0),'Struct_gap_ratio':(5.8,1.75),'Struct_max_gap_area':(7.0,1.1),
       'Text_Entropy':(2.8,4.25),'Text_ASM':(-4.1,-4.35)}
    for feature in ld.index:
        vx,vy=ld.loc[feature,['PC1','PC2']].to_numpy()*scale
        c.annotate('',xy=(vx,vy),xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#666666',lw=.6))
        tx,ty=label_pos[feature]
        c.annotate(feature,xy=(vx,vy),xytext=(tx,ty),textcoords='data',fontsize=6.5,
                   ha='center',va='center',color='#333333',
                   arrowprops=dict(arrowstyle='-',color='#999999',lw=.4,shrinkA=3,shrinkB=2))
    # Keep all 275 projections visible. Original bounds extended only if required.
    xmin=min(-6,float(scores.PC1.min())-.4);xmax=max(8,float(scores.PC1.max())+.4)
    ymin=min(-5,float(scores.PC2.min())-.4);ymax=max(5,float(scores.PC2.max())+.4)
    c.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='PC1 (57.90495% variance)',ylabel='PC2 (17.33318% variance)')
    c.legend(handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#777777',markeredgecolor='white',markersize=4,label='Training (n = 220)'),Line2D([],[],marker='^',color='none',markerfacecolor='#777777',markeredgecolor='white',markersize=4,label='Historical test projection (n = 55)')],loc='lower left',bbox_to_anchor=(0,1.00),fontsize=6.5,handlelength=1)
    cbax2=fig.add_axes([.89,.075,.014,.30]);fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,100),cmap='RdBu_r'),cax=cbax2).set_label('Frozen OCI score',fontsize=7)
    for aa,k in zip([a,b,c],'abc'):letter(aa,k,(-32,8))
    save(fig,'Fig3',[a,b,c],columns=[['a','b','c']],dpi=1000)

def Fig4():
    fr=read('Fig4_selection_frequencies_1000.csv');coef=read('Fig4_model_coefficients.csv').set_index('feature')
    rep=read('Fig4_complete_OOF_repeat_metrics.csv');hold=read('Fig4_historical55_metrics.csv').iloc[0]
    s=read('F2_frozen_scores_275.csv');fig=newfig(222)
    a=axat(fig,[.24,.71,.70,.23],'a');b=axat(fig,[.24,.405,.285,.19],'b');c=axat(fig,[.655,.405,.285,.19],'c')
    d=axat(fig,[.24,.075,.285,.21],'d');e=axat(fig,[.655,.075,.285,.21],'e')
    y=np.arange(len(fr));v=fr.selection_frequency.to_numpy()*100
    a.barh(y,v,color=[RED if q>=70 else '#a7cdd9' for q in v],height=.72)
    a.set_yticks(y,fr.feature,fontsize=7);a.invert_yaxis();a.set(xlim=(0,105),xlabel='Selection frequency across 1000 subsamples (%)')
    a.axvline(70,c='black',ls='--',lw=.8)
    a.text(.96,.02,'Retention threshold: 70%',transform=a.transAxes,ha='right',va='bottom',fontsize=7)
    for yy,q in zip(y,v):a.text(q+.8,yy,f'{q:.1f}%',va='center',fontsize=6.5)
    coefficient_order=['Struct_num_gaps','Color_S','Struct_gap_ratio']
    vals=coef.loc[coefficient_order,'beta_standardized'].to_numpy()
    b.barh(range(3),vals,color=RED,height=.65);b.set_yticks(range(3),coefficient_order,fontsize=7);b.invert_yaxis()
    b.set(xlim=(0,.50),xlabel='Standardized regression coefficient (β)')
    for yy,q in enumerate(vals):b.text(.012,yy,f'{q:.4f}',va='center',color='white',fontsize=7,weight='bold')
    box(c,[rep.R2.to_numpy()],[0],[BLUE],width=.40)
    c.scatter(np.array([-.065,0,.065]),rep.R2,s=14,facecolors=BLUE,edgecolors='white',lw=.25,zorder=4)
    c.scatter([1],[hold.R2],marker='d',s=24,c=ORANGE,edgecolors='white',lw=.4,zorder=4)
    c.axhline(rep.R2.mean(),c=BLUE,ls=':',lw=.8)
    c.set(xlim=(-.5,1.5),ylim=(.20,.65),ylabel='R²')
    c.set_xticks([0,1],['Outer CV\n3 full OOF repeats','Historical test\nn = 55'],fontsize=6.5)
    c.text(.04,.96,f'CV mean {rep.R2.mean():.4f}\nHistorical test {hold.R2:.4f}',transform=c.transAxes,va='top',fontsize=7)
    for aa,target in [(d,'GR'),(e,'DD')]:
        for k,g in enumerate(GRADES):
            z=s[s.OCI_grade_ordinal.eq(k)]
            aa.scatter(z.OCI_Score_recomputed,z[target],s=9,c=COLORS[k],edgecolors='white',lw=.2,alpha=.9,label=g)
        aa.set(xlim=(-2,105),xlabel='OCI score',ylabel='GR (gf s⁻¹)' if target=='GR' else 'DD (mm)',ylim=(0,8000) if target=='GR' else (0,90))
        r=s.OCI_Score_recomputed.corr(s[target]);aa.set_title(f'OCI versus {target} (r = {r:.3f})',pad=7)
    d.legend(loc='upper right',fontsize=6.5,handlelength=1)
    for aa,k in zip([a,b,c,d,e],'abcde'):letter(aa,k,(-32,8))
    save(fig,'Fig4',[a,b,c,d,e],rows=[['b','c'],['d','e']],columns=[['b','d'],['c','e']],dpi=1000)

def Fig5():
    s=read('F2_frozen_scores_275.csv');threshold=jread('D__summary.json')['OCI_grade_score_boundaries_train_only']
    fig=newfig(99);ax=axat(fig,[.12,.21,.82,.65],'a')
    bins=np.linspace(0,100,36)
    values=[s.loc[s.OCI_grade_ordinal.eq(k),'OCI_Score_recomputed'].to_numpy() for k in range(3)]
    ax.hist(values,bins=bins,stacked=True,color=COLORS,edgecolor='white',linewidth=.4,label=GRADES)
    assert sum(np.histogram(s.OCI_Score_recomputed,bins)[0])==275
    for t in threshold:ax.axvline(t,c='#777777',ls='--',lw=.8)
    ax.set(xlim=(0,100),ylim=(0,30),xlabel='OCI score (training-fitted scale)',ylabel='Number of F₂ records')
    for x,g,col,n in [(10.8,'Compact',BLUE,137),(35.6,'Medium','#777777',101),(74.8,'Loose',RED,37)]:
        ax.text(x,28,f'{g} (n = {n})',color=col,weight='bold',ha='center',va='top',fontsize=8)
    ax.text(threshold[0]+1,23,'21.68834',fontsize=7);ax.text(threshold[1]+1,23,'49.63180',fontsize=7)
    ax.set_title('Research-defined OCI categories (n = 275)',pad=10)
    save(fig,'Fig5',[ax],dpi=1000)
    hist=[]
    for k,g in enumerate(GRADES):
        counts,edges=np.histogram(values[k],bins)
        hist.extend(dict(grade=g,bin_left=l,bin_right=u,n=int(n)) for l,u,n in zip(edges[:-1],edges[1:],counts))
    pd.DataFrame(hist).to_csv(D/'Fig5_histogram_counts.csv',index=False)

def flow_patch(ax,x0,x1,y0,y1,h,color,alpha=.55):
    xm=(x0+x1)/2
    vertices=[(x0,y0),(xm,y0),(xm,y1),(x1,y1),(x1,y1+h),(xm,y1+h),(xm,y0+h),(x0,y0+h),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(vertices,codes),fc=color,ec='none',alpha=alpha))

def Fig6():
    m=pd.read_csv(D/'Fig6_contingency_3x3.csv',index_col=0).to_numpy()
    f=read('Fig6_sample_level_275_neutral_labels.csv');fig=newfig(160)
    a=axat(fig,[.10,.57,.35,.33],'a');b=axat(fig,[.61,.57,.29,.33],'b')
    c=axat(fig,[.10,.13,.35,.27],'c');d=axat(fig,[.61,.13,.35,.27],'d')
    im=a.imshow(m,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    a.set_xticks(range(3),GRADES,fontsize=7);a.set_yticks(range(3),GRADES,fontsize=7)
    a.set(xlabel='OCI research grade',ylabel='Mechanical reference grade (TA)');a.tick_params(length=0)
    for i in range(3):
        for k in range(3):a.text(k,i,str(m[i,k]),ha='center',va='center',weight='bold',color='white' if m[i,k]>55 else DARK,fontsize=9)
    a.add_patch(Rectangle((.5,-.5),2,1,fill=False,ec=RED,lw=1))
    a.add_patch(Rectangle((-.5,1.5),2,1,fill=False,ec=BLUE,lw=1))
    ca=fig.add_axes([.465,.57,.012,.33]);fig.colorbar(im,cax=ca).set_label('Records',fontsize=7)
    a.set_title('Agreement 64.0%; unweighted κ = 0.4148',fontsize=7.5,pad=13)
    left=m.sum(1);right=m.sum(0);gap=12;total=275+2*gap
    ly=[];ry=[];cur=total
    for n in left:cur-=n;ly.append(cur);cur-=gap
    cur=total
    for n in right:cur-=n;ry.append(cur);cur-=gap
    lo=np.array(ly,dtype=float)+left;ro=np.array(ry,dtype=float)+right
    flows=[]
    for i in range(3):
        for k in range(3):
            n=m[i,k];lo[i]-=n;ro[k]-=n
            color=RED if i==0 and k>0 else BLUE if i==2 and k<2 else '#c4c7ca'
            flow_patch(b,.08,.92,lo[i],ro[k],n,color,.65 if color!='#c4c7ca' else .40)
            flows.append(dict(TA=GRADES[i],OCI=GRADES[k],n=int(n)))
    for i,col in enumerate(COLORS):
        b.add_patch(Rectangle((0,ly[i]),.08,left[i],fc=col,ec='none'))
        b.add_patch(Rectangle((.92,ry[i]),.08,right[i],fc=col,ec='none'))
        b.text(-.02,ly[i]+left[i]/2,f'TA-{GRADES[i]}\n(n = {left[i]})',ha='right',va='center',fontsize=6.5)
        b.text(1.02,ry[i]+right[i]/2,f'OCI-{GRADES[i]}\n(n = {right[i]})',ha='left',va='center',fontsize=6.5)
    b.set(xlim=(0,1),ylim=(0,total));b.axis('off')
    pd.DataFrame(flows).to_csv(D/'Fig6_Sankey_flows.csv',index=False)
    gc=f[(f.TA_grade=='Compact')&(f.OCI_grade=='Compact')]
    dc=f[(f.TA_grade=='Compact')&(f.OCI_grade!='Compact')]
    assert len(gc)==96 and len(dc)==35
    for k,feature in enumerate(CORE):
        col='Z_'+feature
        box(c,[gc[col].to_numpy(),dc[col].to_numpy()],[k-.18,k+.18],[BLUE,'#b64d58'],width=.29)
        # Deterministic offsets retain every record; no random jitter/subsampling.
        for data,pos,color in [(gc,k-.18,BLUE),(dc,k+.18,'#b64d58')]:
            offsets=((np.arange(len(data))*7)%19-9)/100
            c.scatter(pos+offsets,data[col],s=2,alpha=.35,c=color,edgecolors='none',zorder=3)
    c.set_xticks(range(3),[SHORT[k] for k in CORE]);c.set(xlim=(-.6,2.6),ylim=(-2,5),ylabel='Feature value (training z score)')
    c.axhline(0,c='#888888',ls=':',lw=.5)
    c.legend(handles=[Patch(fc=BLUE,label='TA-Compact / OCI-Compact (n = 96)'),
        Patch(fc='#b64d58',label='TA-Compact / OCI-Medium-or-Loose (n = 35)')],loc='lower left',bbox_to_anchor=(0,1.01),fontsize=6.5)
    bins=np.linspace(2500,8000,21);grid=np.linspace(2500,8000,500)
    for data,color,label in [(gc,BLUE,'Concordant'),(dc,'#b64d58','Discordant')]:
        d.hist(data.GR,bins=bins,density=True,alpha=.42,color=color,ec='white',lw=.3)
        d.plot(grid,gaussian_kde(data.GR)(grid),c=color,lw=.8)
        d.axvline(data.GR.mean(),c=color,ls='--',lw=.8)
    d.set(xlim=(2500,8000),ylim=(0,.0011),xlabel='Force–time gradient GR (gf s⁻¹)',ylabel='Density')
    d.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3))
    # Text is outside the data rectangle to avoid claiming non-significance as equivalence.
    d.text(.02,1.08,f'Mean GR: concordant {gc.GR.mean():.1f}; discordant {dc.GR.mean():.1f}',transform=d.transAxes,fontsize=6.5)
    for aa,k in zip([a,b,c,d],'abcd'):letter(aa,k,(-29,17 if k=='a' else 8))
    save(fig,'Fig6',[a,b,c,d],rows=[['a','b'],['c','d']],columns=[['a','c']],
         exemptions=[{'panels':['b'],'checks':['panel-width'],'reason':'Original Sankey reserves external source/target labels, unlike the heatmap plot area.'}],dpi=1000)

def feature_groups(f2,other):
    model=jread('B__model.json')[MODEL]
    pars={x:(mu,sd) for x,mu,sd in zip(model['features'],model['x_mean'],model['x_sd_ddof1'])}
    return {key:{feature:(df[feature].to_numpy()-pars[feature][0])/pars[feature][1] for feature in CORE}
            for key,df in [('F2',f2),('Other',other)]}

def plot_application(name,other,sensitivity=False):
    s=read('F2_frozen_scores_275.csv');tr=s[s.Set.eq('train')]
    zs=feature_groups(tr,other);fig=newfig(166)
    a=axat(fig,[.11,.58,.35,.31],'a');b=axat(fig,[.11,.12,.35,.31],'b')
    c=axat(fig,[.61,.58,.35,.31],'c');d=axat(fig,[.61,.12,.35,.31],'d')
    display_core=['Struct_num_gaps','Color_S','Struct_gap_ratio']
    for k,f in enumerate(display_core):
        box(a,[zs['F2'][f],zs['Other'][f]],[k-.18,k+.18],[BLUE,ORANGE],width=.30)
    a.set_xticks(range(3),[SHORT[k] for k in display_core]);a.set(xlim=(-.6,2.6),ylim=(-2,5.5),ylabel='Feature value (F₂-training z score)')
    a.axhline(0,c='#777777',ls=':',lw=.5)
    other_label=f'Inbred records (n = {len(other)})'
    a.legend(handles=[Patch(fc=BLUE,label='F₂ training (n = 220)'),Patch(fc=ORANGE,label=other_label)],
             loc='lower left',bbox_to_anchor=(0,1.0),fontsize=6.5)
    c1=[int(tr.OCI_grade_ordinal.eq(k).sum()) for k in range(3)]
    c2=[int(other.OCI_Grade_frozen.eq(g).sum()) for g in GRADES]
    xp=np.arange(3)
    b.bar(xp-.18,np.array(c1)/len(tr)*100,.34,color=BLUE,label='F₂ training')
    b.bar(xp+.18,np.array(c2)/len(other)*100,.34,color=ORANGE,label=other_label)
    for xx,n in zip(xp-.18,c1):b.text(xx,n/len(tr)*100+1,f'{100*n/len(tr):.1f}%',ha='center',va='bottom',color=BLUE,fontsize=6.5)
    for xx,n in zip(xp+.18,c2):b.text(xx,n/len(other)*100+1,f'{100*n/len(other):.1f}%',ha='center',va='bottom',color='#ae622c',fontsize=6.5)
    b.set_xticks(xp,GRADES);b.set(ylim=(0,65),ylabel='Proportion of records (%)')
    # Fixed, common 4-score-unit bins. Include the actual negative inbred score.
    edges=np.arange(-4,105,4)
    c.hist(tr.OCI_Score_recomputed,bins=edges,density=True,alpha=.55,color=BLUE,ec='white',lw=.3,label='F₂ training')
    c.hist(other.OCI_Score_frozen,bins=edges,density=True,alpha=.55,color=ORANGE,ec='white',lw=.3,label=other_label)
    t=jread('D__summary.json')['OCI_grade_score_boundaries_train_only']
    for q in t:c.axvline(q,c='#777777',ls='--',lw=.6)
    c.set(xlim=(-4,104),ylim=(0,.045),xlabel='OCI score (training-fitted scale)',ylabel='Density')
    outside=int(other.any_feature_outside_F2_training_minmax.sum())
    c.text(.97,1.15,f'Outside ≥1 training input range:\n{outside}/{len(other)} records ({100*outside/len(other):.1f}%)',
             transform=c.transAxes,ha='right',va='top',fontsize=6.5)
    # Original panel d uses mean ± SEM within grade. Input-defined gradients are descriptive.
    markers=['o','s','D'];feature_colors=['#3e78a4','#6ba0c7','#a4c4da']
    gradient=[]
    for k,f in enumerate(CORE):
        for dataset,df,color,ls,offset in [('F2',tr,feature_colors[k],'-',-.04),('Other',other,['#c9793e','#e6a16d','#f0c7a8'][k],'--',.04)]:
            temp=df.copy();temp['z']=zs[dataset][f]
            group_col='OCI_grade_ordinal' if dataset=='F2' else 'OCI_Grade_frozen'
            mean=[];sem=[]
            for i,g in enumerate(GRADES):
                q=temp.loc[temp[group_col].eq(i if dataset=='F2' else g),'z']
                mean.append(q.mean());sem.append(q.std(ddof=1)/np.sqrt(len(q)))
                gradient.append(dict(dataset=dataset,feature=f,grade=g,n=len(q),mean_z=q.mean(),SEM=q.std(ddof=1)/np.sqrt(len(q))))
            d.errorbar(xp+offset,mean,yerr=sem,c=color,ls=ls,marker=markers[k],ms=2.7,lw=.85,capsize=1.7,
                       label=('F₂' if dataset=='F2' else 'Inbred')+': '+SHORT[f])
    d.set_xticks(xp,GRADES);d.set(xlim=(-.3,2.3),ylim=(-1.6,4.2),ylabel='Feature mean (training z score)')
    d.axhline(0,c='#888888',ls=':',lw=.5)
    d.legend(loc='lower left',bbox_to_anchor=(-.02,1.01),ncol=2,columnspacing=.5,handlelength=1.3,fontsize=6)
    for aa,k in zip([a,b,c,d],'abcd'):letter(aa,k,(-29,8))
    save(fig,name,[a,b,c,d],rows=[['a','c'],['b','d']],columns=[['a','b'],['c','d']],dpi=1000)
    pd.DataFrame(gradient).to_csv(D/f'{name}_within_grade_means_SEM.csv',index=False,float_format='%.17g')
    hist=[]
    for dataset,v in [('F2 training',tr.OCI_Score_recomputed),('Inbred records',other.OCI_Score_frozen)]:
        n,_=np.histogram(v,edges)
        hist.extend(dict(dataset=dataset,left=l,right=u,n=int(nn)) for l,u,nn in zip(edges[:-1],edges[1:],n))
    pd.DataFrame(hist).to_csv(D/f'{name}_histogram_counts.csv',index=False)

def Fig7():
    # Author-approved primary application set; frozen rows/parameters unchanged.
    other=read('Fig7_inbred_222_records.csv')
    assert len(other)==222 and other.inbred_row_id.is_unique
    assert other.OCI_Grade_frozen.value_counts().reindex(GRADES).tolist()==[102,92,28]
    assert int(other.any_feature_outside_F2_training_minmax.sum())==56
    plot_application('Fig7',other)

def SupplementFig2():
    f=read('F2_all_features_275.csv');cols=['GR','DD']+[x for x in f if x not in ['image_id','GR','DD']]
    stat=read('Supplement_Table1_definitions_statistics.csv').set_index('feature')
    fig=newfig(255);axes=[]
    # Preserve the original six-row layout: two mechanical plots in row one,
    # followed by five rows containing all twenty image variables.
    slots=[(0,0),(0,1)]+[(r,c) for r in range(1,6) for c in range(4)]
    histogram=[]
    for k,(col,(row,column)) in enumerate(zip(cols,slots)):
        left=.085+column*.23;bottom=.075+(5-row)*.15
        ax=axat(fig,[left,bottom,.185,.105],string.ascii_lowercase[k]);axes.append(ax)
        color='#8e91b6' if col in ['GR','DD'] or col.startswith(('Morph_','Size_','Text_')) else '#cb847c' if col.startswith('Color_') else '#91b18a'
        data=f[col].to_numpy();counts,edges=np.histogram(data,bins=20)
        ax.hist(data,bins=edges,color=color,ec='white',lw=.25,alpha=.90)
        grid=np.linspace(data.min(),data.max(),300)
        if np.std(data)>0:ax.plot(grid,gaussian_kde(data)(grid)*len(data)*np.diff(edges).mean(),c=color,lw=.8)
        ax.axvline(np.mean(data),c='#555555',ls=':',lw=.5)
        ax.set_title(col,fontsize=6.5,pad=5);ax.tick_params(labelsize=6)
        # Keep units out of small titles to retain the original feature labels.
        unit=stat.loc[col,'unit']
        if unit in ['pixel^2','pixel','pixel^-1','gf s^-1','mm']:
            ax.set_xlabel(unit.replace('^2','²').replace('^-1','⁻¹').replace('s-1','s⁻¹'),fontsize=6,labelpad=1)
        if max(abs(data.min()),abs(data.max()))>=10000:ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        letter(ax,string.ascii_lowercase[k],(-10,8))
        histogram.extend(dict(feature=col,bin_left=a,bin_right=b,n=int(n)) for a,b,n in zip(edges[:-1],edges[1:],counts))
    fig.text(.5,.98,'Mechanical and image-feature distributions (n = 275)',ha='center',fontsize=9)
    rows=[[string.ascii_lowercase[k] for k,(r,c) in enumerate(slots) if r==row] for row in range(6)]
    columns=[[string.ascii_lowercase[k] for k,(r,c) in enumerate(slots) if c==column] for column in range(4)]
    save(fig,'SupplementFig2',axes,rows=rows,columns=columns,dpi=1000)
    pd.DataFrame(histogram).to_csv(D/'SupplementFig2_histogram_counts.csv',index=False,float_format='%.17g')

def SupplementFig3():
    data=pd.read_csv(I/'D__k_sensitivity.csv');scores=read('F2_frozen_scores_275.csv')
    tr=scores[scores.Set.eq('train')].copy();vals=tr.OCI_Score_recomputed.to_numpy()
    reconstructed=[]
    for _,row in data.iterrows():
        center_strings=row.centers_ascending.split(';')
        centers=np.array([float(v) for v in center_strings])
        labels=np.abs(vals[:,None]-centers).argmin(1)
        # Archived KMeans inertia is relative to its saved centres, which may
        # differ slightly from exact group means after a tolerance stop.
        residual=vals-centers[labels];sse=float(np.sum(residual**2))
        delta=np.array([.5*10**(-len(v.split('.')[1])) for v in center_strings])
        rounding_bound=float(np.sum(2*np.abs(residual)*delta[labels]+delta[labels]**2))
        assert abs(sse-row.training_within_cluster_sse)<rounding_bound+1e-7
        if row.k==3:assert np.array_equal(labels,tr.OCI_grade_ordinal.to_numpy())
        eta={}
        for feature in ['GR','DD']:
            v=tr[feature].to_numpy();ss=((v-v.mean())**2).sum()
            between=sum((labels==k).sum()*(v[labels==k].mean()-v.mean())**2 for k in range(len(centers)))
            eta[feature]=float(between/ss)
        reconstructed.append(dict(k=int(row.k),silhouette=row.training_silhouette_score,
              eta2_GR=eta['GR'],eta2_DD=eta['DD'],SSE_replay=sse,SSE_rounding_error_bound=rounding_bound))
    df=pd.DataFrame(reconstructed);df.to_csv(D/'SupplementFig3_k_sensitivity.csv',index=False,float_format='%.17g')
    fig=newfig(100);a=axat(fig,[.13,.22,.72,.60],'a');b=a.twinx()
    l1=a.plot(df.k,df.silhouette,'o-',color='#a2a2a2',ms=3,lw=1,label='Silhouette score')
    l2=b.plot(df.k,df.eta2_GR,'s-',color=BLUE,ms=3,lw=1,label='Descriptive η² (GR)')
    l3=b.plot(df.k,df.eta2_DD,'^-',color=RED,ms=3,lw=1,label='Descriptive η² (DD)')
    a.axvline(3,c='#777777',ls='--',lw=.65);a.set_xticks(df.k)
    a.set(xlim=(1.8,6.2),ylim=(.45,.72),xlabel='Number of clusters (k)',ylabel='Training silhouette score')
    b.set(ylim=(.20,max(.70,float(df[['eta2_GR','eta2_DD']].max().max())+.03)),ylabel='Descriptive variance explained (η²)')
    lines=l1+l2+l3;a.legend(lines,[l.get_label() for l in lines],loc='lower left',bbox_to_anchor=(0,1.04),ncol=3,fontsize=6.5)
    save(fig,'SupplementFig3',[a],dpi=1000)

def SupplementFig5():
    oof=read('S5_all3_repeat_OOF_predictions.csv');oof=oof[oof.outer_repeat.eq(1)]
    test=read('S5_historical55_predictions.csv');assert len(oof)==220 and len(test)==55
    ci=read('S5_first_repeat_conditional_CI.csv').set_index('metric')
    metrics=read('Fig4_historical55_metrics.csv').iloc[0]
    fig=newfig(105);a=axat(fig,[.12,.22,.35,.65],'a');b=axat(fig,[.61,.22,.35,.65],'b')
    allval=np.r_[oof.y_true,oof.y_pred,test.y_true,test.y_pred]
    low=min(-2,float(allval.min())-.15);high=max(3,float(allval.max())+.15)
    for aa,dd,col,title in [(a,oof,BLUE,'First complete OOF repeat (n = 220)'),(b,test,ORANGE,'Historical test (n = 55)')]:
        aa.scatter(dd.y_true,dd.y_pred,s=12,c=col,ec='white',lw=.3,alpha=.85)
        aa.plot([low,high],[low,high],c='#666666',ls='--',lw=.7)
        aa.set(xlim=(low,high),ylim=(low,high),xlabel='Observed mechanical composite',ylabel='Predicted mechanical composite')
        aa.set_title(title,pad=9,fontsize=7.5)
    a.text(.05,.95,f'R² = {ci.loc["R2","first_repeat_point"]:.4f}\nConditional 95% CI\n[{ci.loc["R2","bootstrap95_low"]:.4f}, {ci.loc["R2","bootstrap95_high"]:.4f}]',transform=a.transAxes,va='top',fontsize=7)
    b.text(.05,.95,f'R² = {metrics.R2:.4f}\nRMSE = {metrics.RMSE:.4f}\nMAE = {metrics.MAE:.4f}',transform=b.transAxes,va='top',fontsize=7)
    letter(a,'a');letter(b,'b');save(fig,'SupplementFig5',[a,b],rows=[['a','b']],dpi=1000)

def export_pdf_asset(doc,name,clip=None,width_mm=190,dpi=600):
    p=figdir(name)
    # Place existing vector/image content at a fixed physical width without rasterizing.
    src=doc[0];rect=fitz.Rect(clip) if clip is not None else src.rect
    width=width_mm*72/25.4;height=width*rect.height/rect.width
    target=fitz.open();pg=target.new_page(width=width,height=height)
    pg.show_pdf_page(pg.rect,doc,0,clip=rect)
    target.save(p/f'{name}.pdf',garbage=4,deflate=True)
    svg=pg.get_svg_image(text_as_path=False)
    # Use the installed family name in editable SVG, not the PDF PostScript alias.
    svg=svg.replace('font-family="TimesNewRomanPSMT"','font-family="Times New Roman"').replace('font-family="TimesNewRomanPS"','font-family="Times New Roman"')
    (p/f'{name}.svg').write_text(svg,encoding='utf-8')
    px=pg.get_pixmap(dpi=dpi,alpha=False)
    im=Image.frombytes('RGB',[px.width,px.height],px.samples)
    im.save(p/f'{name}.tiff',compression='tiff_lzw',dpi=(dpi,dpi))
    pg.get_pixmap(dpi=300,alpha=False).save(p/f'{name}.png')
    write(name+'.retained_asset.json',dict(export_width_mm=width_mm,clip_source_pt=list(rect),
       image_bytes_not_reencoded_in_pdf=True,svg_retains_original_vectors=True,TIFF_is_rendered_copy_not_new_image_resolution=True))
    target.close()

def Fig1():
    source=fitz.open(OUT/'originals/Fig1_V2.pdf');original=source[0]
    # Rebuild only the annotation layer to remove inherited transformed-text
    # boxes. All seven native raster objects and their exact page rectangles
    # are reused, together with the original circular connector geometry.
    drawings=original.get_drawings()
    image_objects=original.get_image_info(xrefs=True)
    spans=[sp for block in original.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for sp in line['spans']]
    width,height=original.rect.width,original.rect.height
    # Reuse image xrefs in the SAME document. Extract/reinsert would convert
    # several CMYK/indexed objects to RGB, contrary to native-asset preservation.
    doc=source;page=doc.new_page(pno=0,width=width,height=height)
    page.insert_font(fontname='TNR',fontfile=FONT_REGULAR)
    tnr=fitz.Font(fontfile=FONT_REGULAR)
    for drawing in drawings:
        if drawing['type']=='s':
            shape=page.new_shape()
            for item in drawing['items']:
                if item[0]=='c':shape.draw_bezier(*item[1:5])
            shape.finish(color=drawing['color'],width=drawing['width']);shape.commit()
        elif drawing['type']=='f' and drawing['rect'].width>200 and drawing['rect'].height>100:
            page.draw_rect(drawing['rect'],color=None,fill=drawing['fill'])
    reused=[]
    for im in image_objects:
        page.insert_image(im['bbox'],xref=im['xref'])
        reused.append(dict(original_xref=im['xref'],source_rectangle=list(im['bbox']),width=im['width'],height=im['height']))
    for drawing in drawings:
        if drawing['type']=='f' and drawing['fill'] and max(drawing['fill'])<.10:
            shape=page.new_shape()
            for item in drawing['items']:
                if item[0]=='c':shape.draw_bezier(*item[1:5])
            shape.finish(color=None,fill=drawing['fill']);shape.commit()
    replacements={'a.Morphology':'a. Morphology','b.Structure':'b. Radial illustration',
                  'c.Color':'c. Color','d.Texture':'d. Texture illustration','e.Size':'e. Size',
                  'b.Structure_Gap':'f. Sectional gaps'}
    changed=[]
    for sp in spans:
        old=sp['text'].strip()
        if old in replacements:
            r=fitz.Rect(sp['bbox']);changed.append((old,replacements[old],list(r)))
    assert len(changed)==6,changed
    for old,new,bbox in changed:
        r=fitz.Rect(bbox);size=10.5
        width=tnr.text_length(new,fontsize=size)
        center=(r.x0+r.x1)/2
        if old=='b.Structure_Gap':center=128
        if old=='b.Structure':center=451
        x=center-width/2
        page.draw_rect(fitz.Rect(x-4,r.y1-13,x+width+4,r.y1+1),fill=(.95,.95,.96),color=None)
        page.insert_text((x,r.y1-2),new,fontname='TNR',fontsize=size,color=(.12,.12,.12))
    for sp in spans:
        if sp['text'].strip() in ['R','G','B','H','S','V']:
            r=fitz.Rect(sp['bbox']);page.insert_text((r.x0,r.y1-2),sp['text'],fontname='TNR',fontsize=8,color=(1,1,1))
    # The size illustration contains two baked annotation strings. Preserve
    # every native image object; cover only their black-background label areas
    # with vector rectangles and reproduce the exact values in embedded TNR.
    size_image=next(im for im in page.get_image_info(xrefs=True) if im['width']==3273 and im['height']==2127)
    r=fitz.Rect(size_image['bbox']);sx=r.width/3273;sy=r.height/2127
    overlays=[]
    for text,box in [
        ('A:4876633',[10,5,750,150]),
        ('D:3052',[10,155,515,290])]:
        patch=fitz.Rect(r.x0+box[0]*sx,r.y0+box[1]*sy,r.x0+box[2]*sx,r.y0+box[3]*sy)
        page.draw_rect(patch,fill=(0,0,0),color=None)
        overlays.append(dict(text=text,source_image_pixels=box,page_rect=list(patch),font='Times New Roman',final_font_pt=6))
    # Place the two unchanged values together in the existing whitespace below
    # the image so a 6pt final font remains legible without touching tissue.
    export_scale=(190*72/25.4)/510
    page.insert_text((r.x0,r.y1+8),'A:4876633; D:3052',fontname='TNR',fontsize=6/export_scale,color=(.12,.12,.12))
    write('Fig1.raster_annotation_font_overlays.json',overlays)
    export_pdf_asset(doc,'Fig1',clip=(45,65,555,610),dpi=600)
    write('Fig1.local_edits.json',dict(replacements=changed,reused_native_images=reused,scientific_images_unmodified=True,
       six_unequal_illustrations_in_original_circular_layout=True,alignment_grid_not_applicable=True))
    doc.close()

def SupplementFig1():
    src=fitz.open(OUT/'originals/Supplement_Figure_Summary.pdf');doc=src
    original=doc[0]
    native=next(im for im in original.get_image_info(xrefs=True) if im['width']==794 and im['height']==477)
    # Reuse the original photographic object on a clean annotation page. This
    # also excludes unused caption fonts from the delivered PDF resources.
    page=doc.new_page(pno=0,width=original.rect.width,height=original.rect.height)
    page.insert_image(native['bbox'],xref=native['xref'],keep_proportion=False)
    image=next(im for im in page.get_image_info(xrefs=True) if im['width']==794 and im['height']==477)
    box=fitz.Rect(image['bbox']);sx=box.width/794;sy=box.height/477
    pix=fitz.Pixmap(doc,image['xref'])
    if pix.colorspace.n!=3:pix=fitz.Pixmap(fitz.csRGB,pix)
    rgb=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
    page.insert_font(fontname='TNRBold',fontfile=FONT_BOLD)
    export_scale=(140*72/25.4)/272
    overlays=[]
    for key,b in zip('abcd',[[126,42,168,71],[417,42,458,72],[127,258,170,294],[417,253,459,289]]):
        arr=rgb[b[1]:b[3],b[0]:b[2]];dark=arr[arr.max(axis=2)<60]
        assert len(dark)>200
        background=tuple((np.median(dark,axis=0)/255).tolist())
        patch=fitz.Rect(box.x0+b[0]*sx,box.y0+b[1]*sy,box.x0+b[2]*sx,box.y0+b[3]*sy)
        page.draw_rect(patch,fill=background,color=None)
        page.insert_text((patch.x0+1/export_scale,patch.y0+10/export_scale),'('+key+')',
            fontname='TNRBold',fontsize=9/export_scale,color=(1,1,1))
        overlays.append(dict(panel=key,source_image_pixels=b,page_rect=list(patch),background_rgb=background,
            font='Times New Roman Bold',final_font_pt=9,scope='Only former letter on non-tissue background; photo object unchanged'))
    export_pdf_asset(doc,'SupplementFig1',clip=(168,90,440,286),width_mm=140,dpi=600)
    write('SupplementFig1.font_overlays.json',overlays)
    write('SupplementFig1.alignment.json',dict(status='RETAINED_PHOTO_PLATE_WITH_TNR_LABELS',
        original_four_photo_quadrants_unchanged=True,plot_area_alignment_not_applicable=True))
    doc.close()

def SupplementFig4():
    # Retain the three matched photographs. Original unresolved d is omitted;
    # 359 remains unchanged in the full 275-record quantitative analyses.
    fig=newfig(85);axes=[]
    records=read('S4_selected_cases_3.csv').set_index('image_id')
    positions=[[.045,.16,.28,.41],[.36,.16,.28,.41],[.675,.16,.28,.41]]
    from PIL import ImageOps
    crops=[]
    for idx,(name,position) in enumerate(zip(['70.jpg','5.jpg','21.jpg'],positions)):
        ax=axat(fig,position,string.ascii_lowercase[idx]);axes.append(ax);ax.axis('off')
        raw=Image.open(OUT/'originals'/name).convert('RGB');arr=np.asarray(raw)
        mask=arr.max(axis=2)>30;yy,xx=np.where(mask);margin=10
        bounds=[max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(raw.width,int(xx.max())+margin+1),min(raw.height,int(yy.max())+margin+1)]
        crop=raw.crop(bounds);canvas=ImageOps.pad(crop,(1600,1050),method=Image.Resampling.LANCZOS,color='black')
        ax.imshow(canvas,aspect='auto');r=records.loc[name]
        ax.text(0,1.07,f'{name} | TA {r.TA_grade}; OCI {r.OCI_grade}\nGR {r.GR:.3f} gf s⁻¹; OCI {r.OCI_Score_recomputed:.3f}\nGaps {int(r.Struct_num_gaps)}; ratio {r.Struct_gap_ratio:.6f}\nColor_S {r.Color_S:.3f}',transform=ax.transAxes,fontsize=6.5,va='bottom')
        crops.append(dict(image=name,display_crop_original_pixels=bounds,colour_adjustment='none',photograph_match='visual same-photograph match against original collage; not an independent field identity log'))
        letter(ax,string.ascii_lowercase[idx],(-10,45))
    save(fig,'SupplementFig4',axes,rows=[['a','b','c']],dpi=600)
    write('SupplementFig4.photo_provenance.json',crops)

FUNCTIONS={n:globals()[n] for n in ['Fig1','Fig2','Fig3','Fig4','Fig5','Fig6','Fig7','SupplementFig1','SupplementFig2','SupplementFig3','SupplementFig4','SupplementFig5']}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('figures',nargs='*');args=parser.parse_args()
    for name in args.figures or list(FUNCTIONS):
        print('Rendering '+name,flush=True);FUNCTIONS[name]()
        print('Completed '+name,flush=True)
