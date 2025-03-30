import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import linregress
from tabulate import tabulate
import numpy as np

# === Load your data here ===
df = pd.read_csv("universities2.csv")

# Filter universities with 10+ years of data
uni_counts = df['name'].value_counts()
eligible_universities = uni_counts[uni_counts >= 10].index
filtered_df = df[df['name'].isin(eligible_universities)]

# === Function ===
def granger_test(university_df):
    university_df = university_df.sort_values('year')
    series = university_df[['teaching_score', 'research_score']].dropna()
    results = {'name': university_df['name'].iloc[0]}
    
    if len(series) > 3:
        for lag in range(1, 3):
            try:
                reversed_series = series[['research_score', 'teaching_score']]
                test_result1 = grangercausalitytests(reversed_series, maxlag=lag, verbose=False)
                f_stat1 = round(test_result1[lag][0]['ssr_ftest'][0], 4)
                p_value1 = round(test_result1[lag][0]['ssr_ftest'][1], 4)
                results[f'rt{lag}_p'] = p_value1
                results[f'rt{lag}_F'] = f_stat1

                test_result2 = grangercausalitytests(series, maxlag=lag, verbose=False)
                f_stat2 = round(test_result2[lag][0]['ssr_ftest'][0], 4)
                p_value2 = round(test_result2[lag][0]['ssr_ftest'][1], 4)
                results[f'tr{lag}_p'] = p_value2
                results[f'tr{lag}_F'] = f_stat2
            except Exception:
                return None
    else:
        return None

    # Avg rank & score
    results['avg_rank'] = round(university_df['rank'].mean(), 2)
    results['avg_score'] = round(university_df['overall_score'].mean(), 2)

    # Trend slope
    if university_df['year'].nunique() > 1:
        slope_rank, _, _, _, _ = linregress(university_df['year'], university_df['rank'])
        slope_score, _, _, _, _ = linregress(university_df['year'], university_df['overall_score'])
        results['slope_rank'] = round(slope_rank, 4)
        results['slope_score'] = round(slope_score, 4)
    else:
        results['slope_rank'] = 'NA'
        results['slope_score'] = 'NA'

    return results

# === Apply ===
result_rows = []
for university in filtered_df['name'].unique():
    uni_df = filtered_df[filtered_df['name'] == university]
    res = granger_test(uni_df)
    if res:
        result_rows.append(res)

result_df = pd.DataFrame(result_rows)

# === Classify Causality ===
def classify(row):
    rt = any(row[f'rt{lag}_p'] < 0.05 for lag in range(1, 3))
    tr = any(row[f'tr{lag}_p'] < 0.05 for lag in range(1, 3))
    if rt and tr:
        return 'Bidirectional'
    elif rt:
        return 'RT Causality'
    elif tr:
        return 'TR Causality'
    else:
        return 'No Causality'

result_df['category'] = result_df.apply(classify, axis=1)

# === Summary with Bidirectional Breakdown ===
summary = []
total = len(result_df)

# RT Causality only
subset_rt = result_df[(result_df['category'] == 'RT Causality')]
count_rt = len(subset_rt)
perc_rt = round(count_rt / total * 100, 1)
f_rt = [v for lag in range(1, 3) for v in subset_rt[f'rt{lag}_F'] if isinstance(v, float)]
avg_rt = round(np.mean(f_rt), 2) if f_rt else '-'
std_rt = round(np.std(f_rt), 2) if f_rt else '-'
summary.append(['RT Causality', count_rt, f"{perc_rt}%", avg_rt, std_rt])

# TR Causality only
subset_tr = result_df[(result_df['category'] == 'TR Causality')]
count_tr = len(subset_tr)
perc_tr = round(count_tr / total * 100, 1)
f_tr = [v for lag in range(1, 3) for v in subset_tr[f'tr{lag}_F'] if isinstance(v, float)]
avg_tr = round(np.mean(f_tr), 2) if f_tr else '-'
std_tr = round(np.std(f_tr), 2) if f_tr else '-'
summary.append(['TR Causality', count_tr, f"{perc_tr}%", avg_tr, std_tr])

# Bidirectional breakdown
subset_bi = result_df[result_df['category'] == 'Bidirectional']
count_bi = len(subset_bi)
perc_bi = round(count_bi / total * 100, 1)
f_bi_rt = [v for lag in range(1, 3) for v in subset_bi[f'rt{lag}_F'] if isinstance(v, float)]
f_bi_tr = [v for lag in range(1, 3) for v in subset_bi[f'tr{lag}_F'] if isinstance(v, float)]
avg_bi = round(np.mean(f_bi_rt + f_bi_tr), 2) if (f_bi_rt + f_bi_tr) else '-'
std_bi = round(np.std(f_bi_rt + f_bi_tr), 2) if (f_bi_rt + f_bi_tr) else '-'
summary.append(['Bidirectional-RT', count_bi, f"{perc_bi}%", round(np.mean(f_bi_rt), 2), round(np.std(f_bi_rt), 2)])
summary.append(['Bidirectional-TR', count_bi, f"{perc_bi}%", round(np.mean(f_bi_tr), 2), round(np.std(f_bi_tr), 2)])

# No Causality
subset_no = result_df[result_df['category'] == 'No Causality']
count_no = len(subset_no)
perc_no = round(count_no / total * 100, 1)
f_no = [v for lag in range(1, 3) for v in subset_no[f'rt{lag}_F'].tolist() + subset_no[f'tr{lag}_F'].tolist() if isinstance(v, float)]
avg_no = round(np.mean(f_no), 2) if f_no else '-'
std_no = round(np.std(f_no), 2) if f_no else '-'
summary.append(['No Causality', count_no, f"{perc_no}%", avg_no, std_no])

# === Final Table Print ===
columns_order = ['name',
                 'rt1_p', 'rt1_F', 'rt2_p', 'rt2_F',
                 'tr1_p', 'tr1_F', 'tr2_p', 'tr2_F',
                 'avg_rank', 'avg_score', 'slope_rank', 'slope_score', 'category']
result_df = result_df[columns_order]
result_df = result_df.sort_values('avg_rank')

print(tabulate(result_df, headers='keys', tablefmt='pretty',
               colalign=("left", "left", "center", "center", "center", "center", "center",
                         "center", "center", "center", "center", "center", "center", "left")))

print("\nSummary of Causality Results:")
print(tabulate(summary, headers=['Category', 'Count', 'Percentage', 'Avg F-stat', 'Std Dev F-stat'],
               tablefmt='pretty', colalign=("left", "center", "center", "center", "center")))
