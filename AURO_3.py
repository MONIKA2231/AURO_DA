pip install pandas prophet sqlalchemy
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def parse_data(text):
    """Parse semi-structured COVID-19 data into a DataFrame"""
    lines = text.strip().split('\n')
    data = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("=====") or stripped.startswith("|---") or stripped == "date location new_cases new_deaths total_cases total_deaths":
            continue

        cleaned = stripped.replace('|', ' ').replace(':', '').replace('  ', ' ').strip()
        tokens = cleaned.split()


        if len(tokens) < 5:
            continue

        date = tokens[0]
        location = tokens[1]

        if location == "United" and len(tokens) > 2 and tokens[2] == "State":
            location = "United States"
            numericals = tokens[3:3+4]
        else:
            numericals = tokens[2:2+4]


        numericals += [None] * (4 - len(numericals))

        try:
            new_cases = int(numericals[0]) if numericals[0] not in [None, ''] else None
            new_deaths = int(numericals[1]) if numericals[1] not in [None, ''] else None
            total_cases = int(numericals[2]) if numericals[2] not in [None, ''] else None
            total_deaths = int(numericals[3]) if numericals[3] not in [None, ''] else None

            data.append([date, location, new_cases, new_deaths, total_cases, total_deaths])
        except:
            continue

    return pd.DataFrame(data, columns=["date", "location", "new_cases", "new_deaths", "total_cases", "total_deaths"])

def clean_data(df):
    """Clean and preprocess the DataFrame"""

    df['date'] = pd.to_datetime(df['date'])

    df['new_cases'] = df['new_cases'].fillna(0)
    df['new_deaths'] = df['new_deaths'].fillna(0)


    df.sort_values(['location', 'date'], inplace=True)
    df['cases_ma7'] = df.groupby('location')['new_cases'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['deaths_ma7'] = df.groupby('location')['new_deaths'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    return df


def analyze_growth_trends(df):
    """Identify growth trends using linear regression"""
    results = []
    locations = df['location'].unique()

    for loc in locations:
        loc_df = df[df['location'] == loc].copy()
        if len(loc_df) < 5:
            continue

        X = np.array(range(len(loc_df))).reshape(-1, 1)
        y_cases = loc_df['cases_ma7'].values


        model = LinearRegression()
        model.fit(X, y_cases)
        case_trend = model.coef_[0]

        # Calculate growth metrics
        peak_cases = loc_df['new_cases'].max()
        peak_cases_date = loc_df.loc[loc_df['new_cases'].idxmax(), 'date'].strftime('%Y-%m-%d')

        try:
            start_case = loc_df['total_cases'].iloc[0]
            end_case = loc_df['total_cases'].iloc[-1]
            n_days = len(loc_df)
            if start_case > 0 and end_case > start_case:
                doubling_time = n_days * np.log(2) / np.log(end_case/start_case)
            else:
                doubling_time = float('inf')
        except:
            doubling_time = float('inf')


        try:
            kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
            cluster_labels = kmeans.labels_
            current_phase = cluster_labels[-1]
        except:
            current_phase = -1

        results.append({
            'location': loc,
            'case_growth_rate': case_trend,
            'peak_cases': peak_cases,
            'peak_cases_date': peak_cases_date,
            'doubling_time': doubling_time,
            'current_phase': current_phase,
            'total_cases': loc_df['total_cases'].iloc[-1],
            'total_deaths': loc_df['total_deaths'].iloc[-1]
        })

    return pd.DataFrame(results)


def generate_summary(analysis_df):
    """Generate a textual summary of key findings"""
    summary = "\n" + "="*80 + "\n"
    summary += "COVID-19 PANDEMIC PATTERN ANALYSIS SUMMARY\n"
    summary += "="*80 + "\n\n"


    world_data = analysis_df[analysis_df['location'] == 'World']
    if not world_data.empty:
        world = world_data.iloc[0]
        summary += "🌍 GLOBAL OVERVIEW:\n"
        summary += f"- Total Cases: {world['total_cases']:,}\n"
        summary += f"- Total Deaths: {world['total_deaths']:,}\n"
        summary += f"- Peak Daily Cases: {world['peak_cases']:,} on {world['peak_cases_date']}\n"
        summary += f"- Current Growth Rate: {world['case_growth_rate']:,.1f} cases/day\n"
        summary += f"- Case Doubling Time: {world['doubling_time']:.1f} days\n\n"
    else:
        summary += "🌍 GLOBAL OVERVIEW: Data for 'World' not available.\n\n"


    fastest_growth = analysis_df[analysis_df['location'] != 'World'].sort_values(
        'case_growth_rate', ascending=False
    ).head(3)

    highest_peaks = analysis_df[analysis_df['location'] != 'World'].sort_values(
        'peak_cases', ascending=False
    ).head(3)

    summary += "🚀 COUNTRIES WITH FASTEST CURRENT GROWTH:\n"
    if not fastest_growth.empty:
        for i, row in fastest_growth.iterrows():
            trend = "↑" if row['case_growth_rate'] > 0 else "↓"
            summary += f"- {row['location']}: {trend} {row['case_growth_rate']:,.1f} cases/day\n"
    else:
        summary += "- No data available for fastest growing countries.\n"

    summary += "\n🔥 COUNTRIES WITH HIGHEST PEAK CASES:\n"
    if not highest_peaks.empty:
        for i, row in highest_peaks.iterrows():
            summary += f"- {row['location']}: {row['peak_cases']:,} cases on {row['peak_cases_date']}\n"
    else:
        summary += "- No data available for countries with highest peak cases.\n"


    summary += "\n\n" + "="*80 + "\n"
    summary += "DETAILED COUNTRY ANALYSIS\n"
    summary += "="*80 + "\n"


    top_countries = analysis_df[analysis_df['location'] != 'World'].sort_values(
        'total_cases', ascending=False
    ).head(10)

    if not top_countries.empty:
        for _, row in top_countries.iterrows():

            if row['current_phase'] == 0:
                phase = "Early Stage"
            elif row['current_phase'] == 1:
                phase = "Growth Phase"
            elif row['current_phase'] == 2:
                phase = "Decline Phase"
            else:
                phase = "Unknown Phase"

            # Determine trend direction
            if row['case_growth_rate'] > 100:
                trend = "Rapid Increase"
            elif row['case_growth_rate'] > 10:
                trend = "Moderate Growth"
            elif row['case_growth_rate'] > 0:
                trend = "Slow Growth"
            elif row['case_growth_rate'] < 0:
                trend = "Declining"
            else:
                trend = "Stable"

            summary += f"\n📍 {row['location'].upper()}:\n"
            summary += f"   - Total Cases: {row['total_cases']:,}\n"
            summary += f"   - Total Deaths: {row['total_deaths']:,}\n"
            summary += f"   - Pandemic Phase: {phase}\n"
            summary += f"   - Trend: {trend}\n"
            summary += f"   - Growth Rate: {row['case_growth_rate']:,.1f} cases/day\n"
            summary += f"   - Peak Cases: {row['peak_cases']:,} on {row['peak_cases_date']}\n"
            summary += f"   - Doubling Time: {row['doubling_time']:.1f} days\n"
    else:
        summary += "- No detailed country data available.\n"

    summary += "\n\n" + "="*80 + "\n"
    summary += "KEY FINDINGS & OBSERVATIONS\n"
    summary += "="*80 + "\n"


    if world_data.empty:
         summary += "\n- Cannot provide global doubling time analysis as 'World' data is missing.\n"
    elif world['doubling_time'] < 10:
        summary += "\n⚠️ GLOBAL WARNING: Cases are doubling rapidly "
        summary += f"(every {world['doubling_time']:.1f} days), indicating uncontrolled spread\n"
    else:
        summary += "\n🌤️ POSITIVE SIGN: Global doubling time is relatively slow "
        summary += f"({world['doubling_time']:.1f} days), suggesting containment measures may be working\n"


    china_data = analysis_df[analysis_df['location'] == 'China']
    italy_data = analysis_df[analysis_df['location'] == 'Italy']
    us_data = analysis_df[analysis_df['location'] == 'United States']

    if not china_data.empty:
        china = china_data.iloc[0]
        if china['case_growth_rate'] < 0:
            summary += f"\n✅ CHINA: Shows declining cases ({china['case_growth_rate']:.1f}/day), "
            summary += "indicating successful containment after initial outbreak\n"

    if not italy_data.empty:
        italy = italy_data.iloc[0]
        if italy['case_growth_rate'] > 500:
            summary += f"\n❗ ITALY: Experiencing rapid growth ({italy['case_growth_rate']:,.1f} cases/day), "
            summary += "suggesting healthcare system may be overwhelmed\n"

    if not us_data.empty:
        us = us_data.iloc[0]
        if us['case_growth_rate'] > 1000:
            summary += f"\n❗ UNITED STATES: Showing explosive growth ({us['case_growth_rate']:,.1f} cases/day), "
            summary += "indicating the outbreak is accelerating\n"
    else:
        summary += "\n- Data for United States is not sufficient for detailed observation.\n"


    summary += "\n\n" + "="*80 + "\n"
    summary += "RECOMMENDATIONS\n"
    summary += "="*80 + "\n"
    summary += "- Countries in growth phase should implement strict containment measures\n"
    summary += "- Nations with rapid doubling times (<7 days) need to increase testing capacity\n"
    summary += "- Regions showing decline should maintain vigilance to prevent resurgence\n"
    summary += "- Global coordination is essential to control cross-border transmission\n"

    return summary

pdf_text = """
date location new_cases new_deaths total_cases total_deaths
1/25/2020 China 462 16 1297 41
1/26/2020 China 688 15 1985 56
1/27/2020 China 776 24 2761 80
1/28/2020 China 1776 26 4537 106
1/29/2020 China 1460 26 5997 132
1/30/2020 China 1739 38 7736 170
1/31/2020 China 1984 43 9720 213
2/1/2020 China 2101 46 11821 259
2/2/2020 China 2590 45 14411 304
2/3/2020 China 2827 57 17238 361
2/4/2020 China 3233 64 20471 425
2/5/2020 China 3892 66 24363 491
2/6/2020 China 3697 73 28060 564
2/7/2020 China 3151 73 31211 637
2/8/2020 China 3387 86 34598 723
2/9/2020 China 2653 89 37251 812
2/10/2020 China 2984 97 40235 909
2/11/2020 China 2473 108 42708 1017
2/12/2020 China 2022 97 44730 1114
2/13/2020 China 1820 254 46550 1368
2/14/2020 China 1998 13 48548 1381
2/15/2020 China 1506 143 50054 1524
2/16/2020 China 1120 142 51174 1666
2/17/2020 China 19461 106 70635 1772
2/18/2020 China 1893 98 72528 1870
2/19/2020 China 1752 136 74280 2006
2/20/2020 China 395 115 74675 2121
2/21/2020 China 894 118 75569 2239
2/22/2020 China 823 109 76392 2348
2/23/2020 China 650 97 77042 2445
2/24/2020 China 220 150 77262 2595
2/25/2020 China 518 71 77780 2666
2/26/2020 China 411 52 78191 2718
2/27/2020 China 439 29 78630 2747
2/28/2020 China 331 44 78961 2791
2/29/2020 China 433 47 79394 2838
3/1/2020 China 574 35 79968 2873
3/2/2020 China 206 42 80174 2915
3/3/2020 China 130 31 80304 2946
3/4/2020 China 118 38 80422 2984
3/5/2020 China 143 31 80565 3015
3/6/2020 China 146 30 80711 3045
3/7/2020 China 102 28 80813 3073
3/8/2020 China 46 27 80859 3100
3/9/2020 China 45 23 80904 3123
3/10/2020 China 20 17 80924 3140
3/11/2020 China 31 22 80955 3162
3/12/2020 China 26 11 80981 3173
3/13/2020 China 10 7 80991 3180
3/14/2020 China 30 14 81021 3194
3/15/2020 China 27 10 81048 3204
3/16/2020 China 29 14 81077 3218
3/17/2020 China 39 13 81116 3231
3/7/2020 France 193 3 613 9
3/8/2020 France 93 1 706 10
3/9/2020 France 410 9 1116 19
3/10/2020 France 286 11 1402 30
3/11/2020 France 372 3 1774 33
3/12/2020 France 495 15 2269 48
3/13/2020 France 591 13 2860 61
3/14/2020 France 780 18 3640 79
3/15/2020 France 829 12 4469 91
3/16/2020 France 911 36 5380 127
3/17/2020 France 1193 21 6573 148
3/10/2020 Germany 27 1139 2
3/11/2020 Germany 157 0 1296 2
3/13/2020 India 1 74 1
3/14/2020 India 8 1 82 2
3/15/2020 India 25 0 107 2
3/16/2020 India 7 0 114 2
3/17/2020 India 23 1 137 3
3/11/2020 International 0 0 696 7
3/12/2020 International 0 0 696 7
3/13/2020 International 0 0 696 7
3/14/2020 International 1 0 697 7
3/15/2020 International 0 0 697 7
3/16/2020 International 15 0 712 7
3/17/2020 International 0 0 712 7
2/20/2020 Iran 2 2
2/21/2020 Iran 3 0 5 2
2/22/2020 Iran 13 2 18 4
2/23/2020 Iran 10 1 28 5
2/24/2020 Iran 15 3 43 8
2/25/2020 Iran 18 4 61 12
2/26/2020 Iran 34 3 95 15
2/27/2020 Iran 46 7 141 22
2/28/2020 Iran 104 4 245 26
2/29/2020 Iran 143 8 388 34
3/1/2020 Iran 205 9 593 43
3/2/2020 Iran 385 11 978 54
3/3/2020 Iran 523 12 1501 66
3/4/2020 Iran 835 11 2336 77
3/5/2020 Iran 586 15 2922 92
3/6/2020 Iran 591 15 3513 107
3/7/2020 Iran 1234 17 4747 124
3/8/2020 Iran 1076 21 5823 145
3/9/2020 Iran 743 49 6566 194
3/10/2020 Iran 595 43 7161 237
3/11/2020 Iran 881 54 8042 291
3/12/2020 Iran 958 63 9000 354
3/13/2020 Iran 1075 75 10075 429
3/14/2020 Iran 1289 85 11364 514
3/15/2020 Iran 1365 94 12729 608
3/16/2020 Iran 2262 245 14991 853
3/17/2020 Iran 0 0 14991 853
3/6/2020 Iraq 0 0 36 2
3/7/2020 Iraq 8 2 44 4
3/8/2020 Iraq 10 0 54 4
3/9/2020 Iraq 6 2 60 6
3/10/2020 Iraq 1 0 61 6
3/11/2020 Iraq 0 0 61 6
3/12/2020 Iraq 9 1 70 7
3/13/2020 Iraq 0 0 70 7
3/14/2020 Iraq 23 2 93 9
3/15/2020 Iraq 0 0 93 9
3/16/2020 Iraq 31 0 124 9
3/17/2020 Iraq 0 0 124 9
3/12/2020 Ireland 9 43 1
3/13/2020 Ireland 27 0 70 1
3/14/2020 Ireland 20 0 90 1
3/15/2020 Ireland 39 1 129 2
3/16/2020 Ireland 40 0 169 2
3/17/2020 Ireland 54 0 223 2
2/23/2020 Italy 67 76 2
2/24/2020 Italy 48 0 124 2
2/25/2020 Italy 105 4 229 6
2/26/2020 Italy 93 5 322 11
2/27/2020 Italy 78 1 400 12
2/28/2020 Italy 250 5 650 17
2/29/2020 Italy 238 4 888 21
3/1/2020 Italy 240 8 1128 29
3/2/2020 Italy 561 6 1689 35
3/3/2020 Italy 347 17 2036 52
3/4/2020 Italy 466 28 2502 80
3/5/2020 Italy 587 27 3089 107
3/6/2020 Italy 769 41 3858 148
3/7/2020 Italy 778 49 4636 197
3/8/2020 Italy 1247 37 5883 234
3/9/2020 Italy 1492 132 7375 366
3/10/2020 Italy 1797 97 9172 463
3/11/2020 Italy 977 168 10149 631
3/12/2020 Italy 2313 196 12462 827
3/13/2020 Italy 2651 189 15113 1016
3/14/2020 Italy 2547 252 17660 1268
3/15/2020 Italy 3497 173 21157 1441
3/16/2020 Italy 3590 368 24747 1809
3/17/2020 Italy 3233 694 27980 2503
3/13/2020 Japan 55 4 675 19
3/14/2020 Japan 41 2 716 21
3/15/2020 Japan 64 1 780 22
3/16/2020 Japan 34 2 814 24
3/17/2020 Japan 15 4 829 28
3/12/2020 Netherlands 121 1 503 5
3/13/2020 Netherlands 111 0 614 5
3/14/2020 Netherlands 190 5 804 10
3/15/2020 Netherlands 155 2 959 12
3/16/2020 Netherlands 176 8 1135 20
3/17/2020 Netherlands 278 4 1413 24
3/14/2020 Philippines 12 0 64 2
3/15/2020 Philippines 47 4 111 6
3/16/2020 Philippines 29 6 140 12
3/17/2020 Philippines 47 0 187 12
3/17/2020 Poland 0 0 150 3
3/14/2020 Spain 1266 36 4231 120
3/15/2020 Spain 1522 16 5753 136
3/16/2020 Spain 2000 152 7753 288
3/17/2020 Spain 1438 21 9191 309
3/14/2020 United States 414 5 1678 41
3/15/2020 United States 0 0 1678 41
3/16/2020 United States 0 0 1678 41
3/17/2020 United States 1825 17 3503 58
2/1/2020 World 2121 46 11953 259
2/2/2020 World 2604 45 14557 305
2/3/2020 World 2834 57 17391 362
2/4/2020 World 3239 64 20630 426
2/5/2020 World 3913 66 24544 492
2/6/2020 World 3712 73 28276 565
2/7/2020 World 3205 73 31481 638
2/8/2020 World 3405 86 34886 724
2/9/2020 World 2672 89 37558 813
2/10/2020 World 2996 97 40554 910
2/11/2020 World 2549 108 43103 1018
2/12/2020 World 2068 97 45171 1115
2/13/2020 World 1826 254 46997 1369
2/14/2020 World 2056 13 49053 1383
2/15/2020 World 1526 143 50580 1526
2/16/2020 World 1277 142 51857 1669
2/17/2020 World 19572 106 71429 1775
2/18/2020 World 1903 98 73332 1873
2/19/2020 World 1872 136 75204 2009
2/20/2020 World 542 115 75748 2129
2/21/2020 World 1021 118 76769 2247
2/22/2020 World 1023 112 77794 2359
2/23/2020 World 1017 102 78811 2463
2/24/2020 World 517 155 79331 2618
2/25/2020 World 896 82 80239 2700
2/26/2020 World 864 62 81109 2762
2/27/2020 World 1175 42 82294 2804
2/28/2020 World 1353 54 83652 2858
2/29/2020 World 1748 66 85403 2924
3/1/2020 World 1727 53 87137 2977
3/2/2020 World 1801 64 88948 3043
3/3/2020 World 1912 67 90869 3112
3/4/2020 World 2217 86 93090 3198
3/5/2020 World 2220 79 95324 3280
3/6/2020 World 2864 99 98192 3380
3/7/2020 World 3730 104 101927 3486
3/8/2020 World 3644 96 105592 3584
3/9/2020 World 3979 224 109577 3809
3/10/2020 World 4119 201 113702 4012
3/11/2020 World 4611 275 118319 4292
3/12/2020 World 6936 317 125260 4613
3/13/2020 World 7488 338 132758 4956
3/14/2020 World 9761 433 142534 5392
3/15/2020 World 10967 343 153517 5735
3/16/2020 World 13971 855 167506 6606
3/17/2020 World 11594 819 179112 7426
"""

print("Processing COVID-19 data...")
df = parse_data(pdf_text)
df = clean_data(df)

print("Analyzing patterns using machine learning...")
analysis_df = analyze_growth_trends(df)

print("Generating summary...")
report = generate_summary(analysis_df)

# Display in terminal
print(report)
print("\nAnalysis complete!")