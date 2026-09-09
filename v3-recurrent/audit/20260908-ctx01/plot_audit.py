from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
snapshot=json.loads((root/'snapshot.json').read_text())
summary=json.loads((root/'summary.json').read_text())
by_name={Path(k.replace('\\','/')).name:v for k,v in summary.items()}
files=['init_masters_forward.jsonl','epoch_1_end_masters_forward.jsonl',
       'checkpoint_epoch_2_batch_2500_masters_forward.jsonl','checkpoint_epoch_2_batch_3000_masters_forward.jsonl']
rows=[by_name[f] for f in files]
steps=[0,4488,6988,7488]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14.2,4.6),layout='constrained')
color='#2166ac'
ss=snapshot['checkpoint_rows']
axes[0].plot([0]+[r['optimizer_step'] for r in ss],[.10009765625]+[r['scale'] for r in ss],'-o',color=color,markersize=4)
axes[0].set(title='Scala appresa',xlabel='Optimizer step',ylabel='context_scale',ylim=(0,.115))
axes[0].annotate('risale a 0,0126',xy=(7488,.0126406),xytext=(3850,.063),
                 arrowprops={'arrowstyle':'->','color':'#555'},fontsize=10)
axes[1].plot(steps,[r['mean_geometry']['raw_context_rms']['mean'] for r in rows],'-o',color='#b2182b')
axes[1].set(title='Contesto prima della LayerNorm',xlabel='Optimizer step',ylabel='RMS media su 80 esempi',ylim=(0,305))
axes[2].plot(steps,[100*r['mean_geometry']['context_between_token_ratio']['mean'] for r in rows],'-o',color='#238b45')
axes[2].set(title='Componente che varia fra token',xlabel='Optimizer step',ylabel='% della RMS del contesto',yscale='log',ylim=(.09,10))
axes[2].set_yticks([.1,1,10],labels=['0,1%','1%','10%'])
for ax in axes:
    ax.grid(alpha=.2)
    ax.set_xticks([0,2000,4000,6000,8000])
fig.suptitle('V3 recurrent: la scala cala mentre il contesto diventa quasi uniforme',fontsize=15)
fig.savefig(root/'context_audit.png',dpi=170,facecolor='white')
plt.close(fig)
