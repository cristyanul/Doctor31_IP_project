import pandas as pd   
import numpy as np  
import plotly.express as px  
from IPython.display import display, clear_output 
from IPython.display import HTML, display 


CSV_PATH = "C:/Users/User/Downloads/doctor31_cazuri(1).csv"  

df = (pd.read_csv(CSV_PATH, parse_dates=['data1'])
        .rename(columns={
            'age_v':'age',
            'greutate':'weight_kg',
            'inaltime':'height_cm',
            'data1':'timestamp',
            'imcINdex':'BMI'
        }))  

print(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols\n")  
display(df.head(10))  

 
df = df.sort_values('timestamp')  

df['dup_within_1h'] = (  
    df.groupby(['age','weight_kg','height_cm'])['timestamp']
      .diff().dt.total_seconds().lt(3600)
).fillna(False)  


df['bmi_invalid'] = (df['BMI'] < 12) | (df['BMI'] > 60)  


df['bmi_cat'] = pd.cut(  
    df['BMI'],  
    bins=[0,18.5,25,30,35,np.inf],  
    labels=['underweight','normal','overweight','obese','extreme_obese']  
)  
df['elderly_obese'] = (df['age']>85) & df['bmi_cat'].isin(['obese','extreme_obese'])  


df['age_invalid']    = df['age']   > 120  
df['weight_invalid'] = (df['weight_kg']<20) | (df['weight_kg']>300)  
df['height_invalid'] = (df['height_cm']<120) | (df['height_cm']>220)  


red_cond    = df[['age_invalid','weight_invalid','height_invalid','bmi_invalid']].any(axis=1)  
yellow_cond = df['dup_within_1h'] | df['elderly_obese']  

df['status'] = np.where(  
    red_cond,  
    'Anomaly',  
    np.where(yellow_cond, 'Suspicious', 'Valid')  
)  

df['color'] = df['status'].map({  
    'Valid':'lightgreen',  
    'Suspicious':'yellow',  
    'Anomaly':'red'  
})  

summary = (df['status']
             .value_counts()
             .reindex(['Valid','Suspicious','Anomaly'], fill_value=0)
             .reset_index(name='count')
             .rename(columns={'index':'status'}))
fig = px.bar(summary, x='status', y='count',
             color='status',
             color_discrete_map={'Valid':'lightgreen','Suspicious':'yellow','Anomaly':'red'},
             title="Doctor31 Data Quality Breakdown")
fig.update_layout(showlegend=False)
fig.show()

def highlight_row(row):
    return [f"background-color: {row.color}" for _ in row.index]

styled = (
    df.style
      .apply(highlight_row, axis=1)
      .format({'BMI':'{:.1f}'})
)