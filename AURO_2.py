!pip install pandas prophet sqlalchemy

import pandas as pd
import re
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def parse_data():
    file_content = """
[file name]: report.pdf
[file content begin]
===== Page 1 =====

Year    Month    Total Number of Calls    Total Number of Doctors Consultancy    Number of Total Health Information

2025 Feb    155200    98964    3777
2025 Jan    130982    87815    3217
2024 Dec    133725    86832    3634
2024 Nov    153698    93710    3010
2024 Oct    214604    108207    6307
2024 Sept    157686    94893    3683
2024 Aug    168418    100902    6435
2024 July    190861    121500    5444
2024 June    169089    111573    6774
2024 May    149502    102415    4982
2024 Apr    159063    109818    5448
2024 Mar    154481    105891    5840
2024 Feb    132505    93147    4486
2024 Jan    158854    113016    4799
2023 Dec    161004    114844    5204
2023 Nov    162097    111330    6555
2023 Oct    205364    131645    9844
2023 Sept    201880    142769    8240
2023 Aug    203224    147923    7233
2023 July    146616    95347    7062
2023 June    125119    77093    6942
2023 May    119520    76269    5610
2023 Apr    116403    75803    5282
2023 Mar    113573    75691    4871
2023 Feb    101276    67396    4594
2023 Jan    98022    64535    4981
2022 Dec    92283    60055    4310
2022 Nov    105178    68375    4821
2022 Oct    124081    80186    6155
2022 Sept    136822    87695    6442
2022 Aug    143461    96837    7335
2022 July    146040    104169    5172
2022 June    177389    118428    7651
2022 May    158578    103472    7353
2022 Apr    161798    103252    7803
2022 Mar    216520    138630    9757
2022 Feb    252123    161242    12787
2022 Jan    280516    184392    13216
2021 Dec    182029    111973    9631
2021 Nov    182042    113548    9080
2021 Oct    227700    136848    12297
2021 Sept    288398    178641    15573

===== Page 2 =====

| 2021 Aug    | 406648    | 249107    | 23167    |
|---|---|---|---|
| 2021 July    | 481426    | 317315    | 22733    |
| 2021 June    | 304620    | 189965    | 15645    |
| 2021 May    | 261009    | 150591    | 15162    |
| 2021 Apr    | 321849    | 203847    | 14749    |
| 2021 Mar    | 258788    | 159314    | 7667    |
| 2021 Feb    | 222444    | 132133    | 8152    |
| 2021 Jan    | 215326    | 116796    | 9038    |
| 2020 Dec    | 217124    | 84535    | 16955    |
| 2020 Nov    | 297991    | 121553    | 24410    |
| 2020 Oct    | 375495    | 163189    | 32425    |
| 2020 Sept    | 437096    | 186377    | 38071    |
| 2020 Aug    | 489118    | 222942    | 42665    |
| 2020 July    | 1481877    | 978094    | 108363    |
| 2020 June    | 2329457    | 1703100    | 173306    |
| 2020 May    | 1972173    | 1522832    | 63111    |
| 2020 Apr    | 2021926    | 1458085    | 103695    |
| 2020 Mar    | 1206116    | 802195    | 96832    |
| 2020 Feb    | 75441    | 38162    | 9845    |
| 2020 Jan    | 74518    | 39216    | 10475    |
| 2019 Dec    | 75181    | 38559    | 10492    |
| 2019 Nov    | 97104    | 51809    | 12112    |
| 2019 Oct    | 95723    | 54081    | 13123    |
| 2019 Sept    | 107937    | 60838    | 14843    |
| 2019 Aug    | 150908    | 76208    | 20186    |
| 2019 July    | 104682    | 57844    | 13305    |
| 2019 June    | 108310    | 58899    | 12937    |
| 2019 May    | 105308    | 60798    | 13205    |
| 2019 Apr    | 107589    | 56656    | 13277    |
| 2019 Mar    | 97763    | 59498    | 12198    |
| 2019 Feb    | 85002    | 53695    | 11028    |
| 2019 Jan    | 85355    | 54771    | 11140    |
| 2018 Dec    | 85514    | 56711    | 10995    |
| 2018 Nov    | 100818    | 66692    | 12117    |
| 2018 Oct    | 124640    | 79248    | 15372    |
| 2018 Sept    | 96148    | 66020    | 11250    |
| 2018 Aug    | 100730    | 67326    | 12982    |
| 2018 July    | 112617    | 79303    | 11839    |
| 2018 June    | 104566    | 76834    | 11755    |
| 2018 May    | 90328    | 62519    | 11515    |
| 2018 Apr    | 84386    | 57144    | 10973    |
| 2018 Mar    | 100355    | 69593    | 12566    |
| 2018 Feb    | 87493    | 59498    | 11518    |
| 2018 Jan    | 69973    | 50665    | 9395    |

===== Page 3 =====

| 2017 Dec    | 75279    | 54367    | 10303    |
|---|---|---|---|
| 2017 Nov    | 78274    | 56864    | 10288    |
| 2017 Oct    | 88852    | 64196    | 11306    |
| 2017 Sept    | 135247    | 98876    | 14984    |
| 2017 Aug    | 209179    | 144087    | 26461    |
| 2017 July    | 84562    | 57021    | 11227    |
| 2017 June    | 83099    | 60437    | 10841    |
| 2017 May    | 72436    | 51541    | 9051    |
| 2017 Apr    | 70740    | 48846    | 9911    |
| 2017 Mar    | 71077    | 47523    | 9893    |
| 2017 Feb    | 65846    | 43645    | 9115    |
| 2017 Jan    | 75328    | 51935    | 9861    |
| 2016 Dec    | 153487    | 114676    | 17823    |
| 2016 Nov    | 130215    | 94646    | 15732    |
| 2016 Oct    | 136981    | 101289    | 15851    |
| 2016 Sept    | 132225    | 101837    | 14408    |
| 2016 Aug    | 142400    | 112831    | 12184    |
| 2016 July    | 145061    | 116686    | 11831    |
| 2016 June    | 186433    | 152580    | 14887    |
| 2016 May    | 166635    | 137326    | 12118    |
| 2016 Apr    | 51432    | 43885    | 3175    |

===== Page 4 =====

Number of Total Ambulance Information    Number of Total Complaints    Number of Calls To Know About The Service

5773    3706    42980
4179    3178    32593
3735    3035    36489
4421    4133    48424
5792    8885    80804
5685    4286    40889
6628    5267    41964
7501    5040    51376
8330    5414    36998
8173    5037    28895
7639    5398    30760
8031    5659    29060
6559    4019    24294
8162    5054    27823
8269    5367    27320
8891    6442    28879
9918    10736    43221
12076    6794    32001
11545    7210    29313
7338    5376    31493
6663    4810    29611
6283    4759    26599
5836    4319    25163
4824    4258    23929
4453    4182    20651
4734    3757    20015
4594    4075    19249
4505    4174    23303
5073    5070    27597
5280    7639    29766
5556    5184    28549
5447    4406    26846
7710    6532    37068
7340    6203    34210
9412    6351    34980
7979    9921    50233
9257    12808    56029
10236    11826    60846
8426    9397    42602
8276    10041    41097
11210    12459    54886
13135    15116    65933

===== Page 5 =====

| 15101    | 22696    | 96577    |
|---|---|---|
| 16566    | 20707    | 104105    |
| 17192    | 11551    | 70267    |
| 17212    | 12106    | 65938    |
| 14905    | 14055    | 74293    |
| 17505    | 13494    | 60808    |
| 16789    | 11497    | 53873    |
| 20738    | 10416    | 58338    |
| 19553    | 10975    | 85106    |
| 31108    | 15993    | 104927    |
| 42829    | 21391    | 115661    |
| 45343    | 25122    | 142183    |
| 47209    | 22663    | 153639    |
| 55451    | 40752    | 299217    |
| 63329    | 47288    | 342434    |
| 58615    | 30241    | 297374    |
| 91751    | 49996    | 318399    |
| 53197    | 59760    | 194132    |
| 628    | 582    | 26224    |
| 650    | 641    | 23536    |
| 768    | 645    | 24717    |
| 933    | 794    | 31456    |
| 1074    | 891    | 26554    |
| 1258    | 1081    | 29917    |
| 1859    | 1692    | 50963    |
| 921    | 871    | 31741    |
| 944    | 988    | 34542    |
| 911    | 829    | 29565    |
| 914    | 976    | 35766    |
| 808    | 800    | 24459    |
| 637    | 676    | 18966    |
| 671    | 578    | 18195    |
| 636    | 551    | 16621    |
| 758    | 781    | 20470    |
| 1065    | 1007    | 27948    |
| 788    | 729    | 17361    |
| 857    | 700    | 18865    |
| 816    | 723    | 19936    |
| 689    | 788    | 14500    |
| 704    | 964    | 14626    |
| 599    | 742    | 14928    |
| 578    | 777    | 16841    |
| 592    | 694    | 15191    |
| 353    | 373    | 9187    |

===== Page 6 =====

| 368    | 469    | 9772    |
|---|---|---|
| 352    | 474    | 10296    |
| 575    | 593    | 12182    |
| 1500    | 1230    | 18657    |
| 3649    | 2741    | 32241    |
| 395    | 512    | 15407    |
| 334    | 327    | 11160    |
| 379    | 394    | 11071    |
| 422    | 488    | 11073    |
| 603    | 590    | 12468    |
| 668    | 528    | 11890    |
| 635    | 525    | 12372    |
| 996    | 294    | 19698    |
| 1144    | 358    | 18335    |
| 1075    | 350    | 18416    |
| 1038    | 412    | 14530    |
| 980    | 340    | 16065    |
| 922    | 404    | 15218    |
| 856    | 473    | 17637    |
| 925    | 439    | 15827    |
| 220    | 77    | 4075    |


[file content end]
"""

    lines = file_content.strip().split('\n')
    main_data = []
    additional_data = []
    in_main = False
    in_additional = False

    month_map = {
        "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr",
        "May": "May", "Jun": "Jun", "Jul": "Jul", "Aug": "Aug",
        "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dec",
        "Sept": "Sep", "July": "Jul", "June": "Jun"
    }

    for line in lines:
        if "===== Page 1 =====" in line:
            in_main = True
            continue
        elif "===== Page 4 =====" in line:
            in_main = False
            in_additional = True
            continue
        elif "===== Page 5 =====" in line or "===== Page 6 =====" in line:
            continue

        if in_main and line.strip() and not line.startswith("Year") and not line.startswith("====="):

            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 4:
                    year_month = parts[0].split()
                    if len(year_month) == 2:
                        year, month = year_month
                        month = month_map.get(month, month)
                        try:
                            main_data.append({
                                'Year': int(year),
                                'Month': month,
                                'Total_Calls': int(parts[1].replace(',', '')),
                                'Doctor_Consultancy': int(parts[2].replace(',', '')),
                                'Health_Information': int(parts[3].replace(',', ''))
                            })
                        except ValueError:
                            continue
            else:

                clean_line = re.sub(r'\s+', ' ', line.strip())
                parts = clean_line.split(' ')

                if len(parts) >= 5:
                    year = parts[0]
                    month = parts[1]
                    month = month_map.get(month, month)
                    try:
                        main_data.append({
                            'Year': int(year),
                            'Month': month,
                            'Total_Calls': int(parts[2].replace(',', '')),
                            'Doctor_Consultancy': int(parts[3].replace(',', '')),
                            'Health_Information': int(parts[4].replace(',', ''))
                        })
                    except ValueError:
                        continue

        if in_additional and line.strip() and not line.startswith("Number of") and not line.startswith("====="):
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 3:
                    try:
                        additional_data.append({
                            'Ambulance_Info': int(parts[0].replace(',', '')),
                            'Complaints': int(parts[1].replace(',', '')),
                            'Service_Calls': int(parts[2].replace(',', ''))
                        })
                    except ValueError:
                        continue
            else:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 3:
                    try:
                        additional_data.append({
                            'Ambulance_Info': int(parts[0].replace(',', '')),
                            'Complaints': int(parts[1].replace(',', '')),
                            'Service_Calls': int(parts[2].replace(',', ''))
                        })
                    except ValueError:
                        continue


    combined_data = []
    min_length = min(len(main_data), len(additional_data))
    for i in range(min_length):
        combined_row = {**main_data[i], **additional_data[i]}
        combined_data.append(combined_row)

    return combined_data

Base = declarative_base()

class CallData(Base):
    __tablename__ = 'call_data'
    id = Column(Integer, primary_key=True)
    year = Column(Integer)
    month = Column(String)
    total_calls = Column(Integer)
    doctor_consultancy = Column(Integer)
    health_information = Column(Integer)
    ambulance_info = Column(Integer)
    complaints = Column(Integer)
    service_calls = Column(Integer)
    date = Column(Date)

def insert_data(data):
    engine = create_engine('sqlite:///call_data.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        month_num_map = {
            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
            "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
            "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
        }

        for row in data:
            month_name = row['Month']
            month_num = month_num_map.get(month_name, "01")
            date_str = f"{row['Year']}-{month_num}-01"
            date = datetime.strptime(date_str, '%Y-%m-%d').date()

            record = CallData(
                year=row['Year'],
                month=row['Month'],
                total_calls=row['Total_Calls'],
                doctor_consultancy=row['Doctor_Consultancy'],
                health_information=row['Health_Information'],
                ambulance_info=row['Ambulance_Info'],
                complaints=row['Complaints'],
                service_calls=row['Service_Calls'],
                date=date
            )
            session.add(record)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

    return engine


def forecast_calls(engine, periods=12):
    query = "SELECT date, total_calls FROM call_data ORDER BY date"
    df = pd.read_sql(query, engine)
    df.rename(columns={'date': 'ds', 'total_calls': 'y'}, inplace=True)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    return model, forecast


def analyze_and_predict(engine, forecast_df, model):

    historical_query = "SELECT * FROM call_data ORDER BY date"
    historical = pd.read_sql(historical_query, engine)


    forecast_df = forecast_df.tail(12).copy()
    forecast_df['Type'] = 'Forecast'
    forecast_df['date'] = forecast_df['ds']


    combined = pd.concat([
        historical[['date', 'total_calls', 'doctor_consultancy',
                   'health_information', 'ambulance_info',
                   'complaints', 'service_calls']],
        forecast_df[['date', 'yhat']].rename(columns={'yhat': 'total_calls'})
    ], ignore_index=True)
    combined['Type'] = combined['date'].apply(lambda x: 'Historical' if x in historical['date'].values else 'Forecast')


    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m')

    combined.to_csv('call_forecast_results.csv', index=False)

    files.download('call_forecast_results.csv')

    fig = model.plot(forecast_df)
    plt.title('Call Volume Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Calls')
    plt.show()

    fig2 = model.plot_components(forecast_df)
    plt.show()

    print("\nForecast for the next 12 months:")
    forecast_df = forecast_df[['ds', 'yhat']].tail(12)
    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m')
    forecast_df.columns = ['Date', 'Predicted Calls']
    print(forecast_df.to_string(index=False))

    return combined

if __name__ == "__main__":

    combined_data = parse_data()

    engine = insert_data(combined_data)

    model, forecast_df = forecast_calls(engine)

    results = analyze_and_predict(engine, forecast_df, model)
    print("\nAnalysis complete! Results saved to 'call_forecast_results.csv'")