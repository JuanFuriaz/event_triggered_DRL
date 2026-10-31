import csv
from decimal import Decimal
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Independent decimal validation of descriptive plotted values')
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent.parent/'tables_mv2boll')
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent.parent/'imgs/mv2boll/descriptive')
args=parser.parse_args()
base=args.data_dir
raw=list(csv.DictReader((base/'metrics_table.csv').open()))
plotted=list(csv.DictReader((args.output_dir/'descriptive_performance_values.csv').open()))
datasets=['mvx2','mvx2_trg','bollx','bollx_trg']
metrics=['Return%_mean','Sharpe_mean','MaxDD%_mean','Trades_mean']
for r in plotted:
 ds=['all'] if r['schedule']=='Daily DRL benchmark' else [(('flt_' if r['schedule']=='Crossover scheduling' else '')+d) for d in datasets]
 selected=[x for x in raw if x['dataset'] in ds]
 if r['block']=='Algorithm':selected=[x for x in selected if x['algo']==r['row'].lower()]
 if r['block']=='Evaluation window':selected=[x for x in selected if x['window']=={'2020':'w1','2022':'w4','2025':'w3'}[r['row']]]
 if r['block']=='Indicator family' and r['schedule']!='Daily DRL benchmark':
  selected=[x for x in selected if ('mv' in x['dataset'])==(r['row']=='Moving average')]
 if r['block']=='Representation' and r['schedule']!='Daily DRL benchmark':
  selected=[x for x in selected if x['dataset'].endswith('_trg')==(r['row']=='Crossovers')]
 assert len(selected)==int(r['n_configurations'])
 for metric in metrics:
  exact=sum(Decimal(x[metric]) for x in selected)/Decimal(len(selected))
  assert abs(exact-Decimal(r[metric]))<Decimal('1e-10'),(r['row'],metric)
retention=list(csv.DictReader((args.output_dir/'descriptive_input_retention_values.csv').open()))
expected_counts={'flt_mvx2':(1692,181),'flt_mvx2_trg':(1692,121),
                 'flt_bollx':(2731,181),'flt_bollx_trg':(2731,121)}
assert len(retention)==4 and {r['dataset'] for r in retention}==set(expected_counts)
for row in retention:
 assert (int(row['dates']),int(row['state_dimension']))==expected_counts[row['dataset']]
 assert (int(row['baseline_dates']),int(row['baseline_state_dimension']))==(4275,361)
 for window in ['w1','w4','w3']:
  dates={r['Dataset']:r for r in csv.DictReader((base/window/'data_reduction.csv').open())}
  states={r['Dataset']:r for r in csv.DictReader((base/window/'state_space_reduction.csv').open())}
  assert int(dates['all']['Timesteps'])==4275
  assert int(dates[row['dates_source']]['Timesteps'])==int(row['dates'])
  assert int(states[row['state_source']]['State_Space'])==int(row['state_dimension'])
 t=Decimal(row['dates'])/Decimal(4275)
 d=Decimal(row['state_dimension'])/Decimal(361)
 for key,exact in [('retained_pct',100*t*d),('date_retention_pct',100*t),('state_retention_pct',100*d)]:
  assert abs(exact-Decimal(row[key]))<Decimal('1e-10'),(row['dataset'],key)
 assert abs(Decimal(row['date_retention_pct'])*Decimal(row['state_retention_pct'])/100
            -Decimal(row['retained_pct']))<Decimal('1e-10')
mean_retention=sum(Decimal(r['retained_pct']) for r in retention)/4
assert abs(mean_retention-Decimal('21.638172069096'))<Decimal('1e-10')
assert [round(Decimal(r['retained_pct']),2) for r in retention]==list(map(Decimal,['19.84','13.27','32.03','21.41']))
for dim,expected in [(181,'50.14'),(121,'33.52'),(361,'100.00')]:
 assert round(100*Decimal(dim)/Decimal(361),2)==Decimal(expected)
print('Independent csv/Decimal validation: all 132 performance values and four retention ratios agree within 1e-10.')
print('Validated date/state-width factors against all three raw exports, multiplicative identity, pooled mean and daily/benchmark retention.')
lookup={(r['row'],r['schedule']):r for r in plotted}
for r in csv.DictReader((args.output_dir/'descriptive_performance_changes.csv').open()):
 for metric in metrics:
  expected=Decimal(lookup[r['row'],'Crossover scheduling'][metric])-Decimal(lookup[r['row'],'Matched daily'][metric])
  assert abs(expected-Decimal(r[metric]))<Decimal('1e-10')
print('Validated 44 unchanged performance differences.')

for r in csv.DictReader((args.output_dir/"descriptive_performance_changes.csv").open()):
 assert abs(Decimal(r["Trades_avoided"])+Decimal(r["Trades_mean"]))<Decimal("1e-10")
print("Validated all eleven Trades avoided sign reversals.")
