"""Descriptive crossover scheduling versus broad daily DRL benchmark.
Preserves the separate matched-DD figure. Run: python descriptive_benchmark_figure.py
"""
from pathlib import Path
import argparse,csv,json,hashlib
from decimal import Decimal
import matplotlib.pyplot as plt
from utils.descriptive_figures import performance_data,performance_changes,plot_performance,METRICS

def generate(data_dir,output_dir):
 data_dir,output_dir=Path(data_dir),Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 summary,_=performance_data(data_dir)
 # The plotting helper expects a reference called Matched daily internally.
 # This local frame replaces it explicitly; no source or matched output is changed.
 frame=summary.loc[summary.schedule.ne('Matched daily')].copy()
 frame['schedule']=frame.schedule.replace({'Daily DRL benchmark':'Matched daily'})
 changes=performance_changes(frame)
 changes['reference']='Daily DRL benchmark (all)'
 raw=list(csv.DictReader((data_dir/'metrics_table.csv').open()))
 for _,r in changes.iterrows():
  scheduled=[x for x in raw if x['dataset'] in ['flt_mvx2','flt_mvx2_trg','flt_bollx','flt_bollx_trg']]
  baseline=[x for x in raw if x['dataset']=='all']
  if r['block']=='Algorithm':
   scheduled=[x for x in scheduled if x['algo']==r['row'].lower()];baseline=[x for x in baseline if x['algo']==r['row'].lower()]
  elif r['block']=='Evaluation window':
   w={'2020':'w1','2022':'w4','2025':'w3'}[r['row']]
   scheduled=[x for x in scheduled if x['window']==w];baseline=[x for x in baseline if x['window']==w]
  elif r['block']=='Representation':scheduled=[x for x in scheduled if x['dataset'].endswith('_trg')==(r['row']=='Crossovers')]
  elif r['block']=='Indicator family':scheduled=[x for x in scheduled if ('mv' in x['dataset'])==(r['row']=='Moving average')]
  for m in METRICS:
   delta=sum(Decimal(x[m]) for x in scheduled)/len(scheduled)-sum(Decimal(x[m]) for x in baseline)/len(baseline)
   assert abs(delta-Decimal(str(r[m])))<Decimal('1e-10')
  assert abs(r['Trades_avoided']+r['Trades_mean'])<1e-10
 changes.to_csv(output_dir/'descriptive_vs_daily_benchmark_values.csv',index=False)
 fig=plot_performance(frame)
 for ext in ['png','pdf','svg']:fig.savefig(output_dir/f'descriptive_vs_daily_benchmark.{ext}',dpi=180,facecolor='white')
 plt.close(fig)
 notes='Descriptive equal-weight differences of per-seed metric means: crossover scheduling minus the broad daily DRL benchmark (dataset all). Trades avoided reverses the subtraction. Algorithm rows use the same algorithm benchmark averaged over three windows; window rows use the same window benchmark averaged over three algorithms. Overall, family and representation rows use the equally weighted benchmark average over all nine algorithm-window cells. These comparisons change representation and scheduling together and do not isolate scheduling or establish significance. The matched-DD figure is preserved separately.\n'
 (output_dir/'descriptive_vs_daily_benchmark_caption.txt').write_text(notes)
 (output_dir/'descriptive_vs_daily_benchmark_provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256((data_dir/'metrics_table.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'validation':'44 differences independently reproduced from CSV with Decimal; 11 trade-sign reversals checked','notes':notes},indent=2)+'\n')
 return changes
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent.parent/'tables_mv2boll');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent.parent/'imgs/mv2boll/descriptive');a=p.parse_args();r=generate(a.data_dir,a.output_dir);print(r[['row']+METRICS[:3]+['Trades_avoided']].round(4).to_string(index=False))
