import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import linregress
from tabulate import tabulate

# Load data
df = pd.read_csv("universities2.csv")

# Filter universities with 10+ years of data
uni_counts = df['name'].value_counts()
eligible_universities = uni_counts[uni_counts >= 10].index
filtered_df = df[df['name'].isin(eligible_universities)]

# Function to calculate Granger causality (lag 1 & 2) + avg & slope
def granger_test(university_df):
    university_df = university_df.sort_values('year')
    series = university_df[['teaching_score', 'research_score']].dropna()
    results = {'name': university_df['name'].iloc[0]}
    
    if len(series) > 3:
        for lag in range(1, 3):
            try:
                # Research -> Teaching
                reversed_series = series[['research_score', 'teaching_score']]
                test_result1 = grangercausalitytests(reversed_series, maxlag=lag, verbose=False)
                f_stat1 = round(test_result1[lag][0]['ssr_ftest'][0], 4)
                p_value1 = round(test_result1[lag][0]['ssr_ftest'][1], 4)
                results[f'rt{lag}_p'] = p_value1
                results[f'rt{lag}_F'] = f_stat1

                # Teaching -> Research
                test_result2 = grangercausalitytests(series, maxlag=lag, verbose=False)
                f_stat2 = round(test_result2[lag][0]['ssr_ftest'][0], 4)
                p_value2 = round(test_result2[lag][0]['ssr_ftest'][1], 4)
                results[f'tr{lag}_p'] = p_value2
                results[f'tr{lag}_F'] = f_stat2

            except Exception:
                results[f'rt{lag}_p'] = 'Error'
                results[f'rt{lag}_F'] = 'Error'
                results[f'tr{lag}_p'] = 'Error'
                results[f'tr{lag}_F'] = 'Error'
    else:
        for lag in range(1, 3):
            results[f'rt{lag}_p'] = 'Insufficient'
            results[f'rt{lag}_F'] = 'Insufficient'
            results[f'tr{lag}_p'] = 'Insufficient'
            results[f'tr{lag}_F'] = 'Insufficient'

    # Add average rank and score
    results['avg_rank'] = round(university_df['rank'].mean(), 2)
    results['avg_score'] = round(university_df['overall_score'].mean(), 2)

    # Add trend slope
    if university_df['year'].nunique() > 1:
        slope_rank, _, _, _, _ = linregress(university_df['year'], university_df['rank'])
        slope_score, _, _, _, _ = linregress(university_df['year'], university_df['overall_score'])
        results['slope_rank'] = round(slope_rank, 4)
        results['slope_score'] = round(slope_score, 4)
    else:
        results['slope_rank'] = 'NA'
        results['slope_score'] = 'NA'

    return results

# Apply and filter
result_rows = []
for university in filtered_df['name'].unique():
    uni_df = filtered_df[filtered_df['name'] == university]
    granger_results = granger_test(uni_df)

    # Filter valid numeric results
    values = list(granger_results.values())
    if all(isinstance(v, (float, int)) for v in values[1:9]):  # exclude name, check Granger fields
        result_rows.append(granger_results)

# Create DataFrame
result_df = pd.DataFrame(result_rows)

# Reorder columns
columns_order = ['name',
                 'rt1_p', 'rt1_F', 'rt2_p', 'rt2_F',
                 'tr1_p', 'tr1_F', 'tr2_p', 'tr2_F',
                 'avg_rank', 'avg_score', 'slope_rank', 'slope_score']
result_df = result_df[columns_order]

# Sort by avg_rank ascending (best rank first)
result_df = result_df.sort_values('avg_rank')

# Print neatly
print(tabulate(result_df, headers='keys', tablefmt='pretty',
               colalign=("left", "left", "center", "center", "center", "center", "center",
                         "center", "center", "center", "center", "center", "center")))

# Optional: Save
result_df.to_csv("granger_results_n2_with_F_trend_sorted.csv", index=False)
print("\nResults saved to granger_results_n2_with_F_trend_sorted.csv")
