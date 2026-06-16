import pandas as pd
df = pd.read_csv('sample_data/contracts_sample.csv', encoding='utf-8-sig')
print('总行数:', len(df))
print('列名:', list(df.columns))
print()
empty_text = df['条款文本'].isna() | (df['条款文本'].astype(str).str.strip() == '')
print('空文本行:', empty_text.sum())
print('  行号:', list(df[empty_text].index + 1))

empty_label = df['风险标签'].isna() | (df['风险标签'].astype(str).str.strip() == '')
print('空标签行:', empty_label.sum())
print('  行号:', list(df[empty_label].index + 1))

print()
print('标签分布:')
print(df['风险标签'].value_counts(dropna=False))
