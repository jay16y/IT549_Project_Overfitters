# enrich_metadata.py
import pandas as pd

meta  = pd.read_csv('models/index_metadata.csv')[['index_position','pill_id']]
ref   = pd.read_csv('models/reference_mapping.csv')[['pill_id','drug_name','ndc_clean']]
pills = ref.drop_duplicates('pill_id')

extra = pd.read_csv('models/pill_metadata.csv')
pills = pills.merge(extra, on='ndc_clean', how='left')

meta  = meta.merge(pills, on='pill_id', how='left')
meta.to_csv('models/index_metadata.csv', index=False)
print(meta[['drug_name','shape','colors','imprint']].head(5))
print('Non-null shapes:', meta['shape'].notna().sum())