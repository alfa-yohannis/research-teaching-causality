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

    results['avg_rank'] = round(university_df['rank'].mean(), 2)
    results['avg_score'] = round(university_df['overall_score'].mean(), 2)

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
def classify(row, lag):
    rt = row[f'rt{lag}_p'] < 0.05
    tr = row[f'tr{lag}_p'] < 0.05
    if rt and tr:
        return 'Bidirectional'
    elif rt:
        return 'RT Causality'
    elif tr:
        return 'TR Causality'
    else:
        return 'No Causality'

result_df['category_n1'] = result_df.apply(lambda r: classify(r, 1), axis=1)
result_df['category_n2'] = result_df.apply(lambda r: classify(r, 2), axis=1)

def classify_combined(row):
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

result_df['category_combined'] = result_df.apply(classify_combined, axis=1)

# === Summary Function ===
def summarize(df, category_col):
    summary = []
    total = len(df)

    for cat in ['RT Causality', 'TR Causality', 'Bidirectional', 'No Causality']:
        subset = df[df[category_col] == cat]
        count = len(subset)
        perc = round(count / total * 100, 1)

        if cat != 'Bidirectional':
            f_values = []
            if category_col == 'category_combined':
                for lag in range(1, 3):
                    f_values += [v for v in subset[f'rt{lag}_F'].tolist() + subset[f'tr{lag}_F'].tolist() if isinstance(v, float)]
            else:
                lag = int(category_col[-1])
                f_values += [v for v in subset[f'rt{lag}_F'].tolist() + subset[f'tr{lag}_F'].tolist() if isinstance(v, float)]

            avg_f = round(np.mean(f_values), 2) if f_values else '-'
            std_f = round(np.std(f_values), 2) if f_values else '-'
            med_f = round(np.median(f_values), 2) if f_values else '-'
            min_f = round(np.min(f_values), 2) if f_values else '-'
            max_f = round(np.max(f_values), 2) if f_values else '-'

            summary.append([cat, count, f"{perc}%", avg_f, std_f, med_f, min_f, max_f])
        else:
            # Bidirectional overall
            count_bi = len(subset)
            perc_bi = round(count_bi / total * 100, 1)
            f_values = []
            if category_col == 'category_combined':
                for lag in range(1, 3):
                    f_values += [v for v in subset[f'rt{lag}_F'].tolist() + subset[f'tr{lag}_F'].tolist() if isinstance(v, float)]
            else:
                lag = int(category_col[-1])
                f_values += [v for v in subset[f'rt{lag}_F'].tolist() + subset[f'tr{lag}_F'].tolist() if isinstance(v, float)]
            avg_f = round(np.mean(f_values), 2) if f_values else '-'
            std_f = round(np.std(f_values), 2) if f_values else '-'
            med_f = round(np.median(f_values), 2) if f_values else '-'
            min_f = round(np.min(f_values), 2) if f_values else '-'
            max_f = round(np.max(f_values), 2) if f_values else '-'
            summary.append(['Bidirectional', count_bi, f"{perc_bi}%", avg_f, std_f, med_f, min_f, max_f])

            # Bidirectional RT and TR
            f_bi_rt = []
            f_bi_tr = []
            if category_col == 'category_combined':
                for lag in range(1, 3):
                    f_bi_rt += [v for v in subset[f'rt{lag}_F'] if isinstance(v, float)]
                    f_bi_tr += [v for v in subset[f'tr{lag}_F'] if isinstance(v, float)]
            else:
                lag = int(category_col[-1])
                f_bi_rt = [v for v in subset[f'rt{lag}_F'] if isinstance(v, float)]
                f_bi_tr = [v for v in subset[f'tr{lag}_F'] if isinstance(v, float)]

            summary.append(['Bidirectional-RT', count_bi, f"{perc_bi}%",
                            round(np.mean(f_bi_rt), 2) if f_bi_rt else '-',
                            round(np.std(f_bi_rt), 2) if f_bi_rt else '-',
                            round(np.median(f_bi_rt), 2) if f_bi_rt else '-',
                            round(np.min(f_bi_rt), 2) if f_bi_rt else '-',
                            round(np.max(f_bi_rt), 2) if f_bi_rt else '-'])

            summary.append(['Bidirectional-TR', count_bi, f"{perc_bi}%",
                            round(np.mean(f_bi_tr), 2) if f_bi_tr else '-',
                            round(np.std(f_bi_tr), 2) if f_bi_tr else '-',
                            round(np.median(f_bi_tr), 2) if f_bi_tr else '-',
                            round(np.min(f_bi_tr), 2) if f_bi_tr else '-',
                            round(np.max(f_bi_tr), 2) if f_bi_tr else '-'])

    return summary

# === Generate Summaries ===
summary_n1 = summarize(result_df, 'category_n1')
summary_n2 = summarize(result_df, 'category_n2')
summary_combined = summarize(result_df, 'category_combined')

# === Final Table Print ===
columns_order = ['name',
                 'rt1_p', 'rt1_F', 'rt2_p', 'rt2_F',
                 'tr1_p', 'tr1_F', 'tr2_p', 'tr2_F',
                 'avg_rank', 'avg_score', 'slope_rank', 'slope_score', 'category_combined']
result_df = result_df[columns_order]
result_df = result_df.sort_values('avg_rank')

print("\nDetailed Causality Results Table:")
print(tabulate(result_df, headers='keys', tablefmt='pretty',
               colalign=("left", "left", "center", "center", "center", "center", "center",
                         "center", "center", "center", "center", "center", "center", "left")))

# === Save to CSV ===
result_df.to_csv("granger_detailed_table.csv", index=False)
print("\nDetailed results table saved to granger_detailed_table.csv")

# === Print Summary Tables ===
headers = ['Category', 'Count', 'Percentage',
           'Avg F', 'Std F', 'Med F', 'Min F', 'Max F']

print("\nSummary of Causality Results (n=1):")
print(tabulate(summary_n1, headers=headers, tablefmt='pretty'))

print("\nSummary of Causality Results (n=2):")
print(tabulate(summary_n2, headers=headers, tablefmt='pretty'))

print("\nSummary of Causality Results (Combined n=1 & n=2):")
print(tabulate(summary_combined, headers=headers, tablefmt='pretty'))
