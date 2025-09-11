import pandas as pd
import os
import requests

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"
df1 = pd.read_csv(data_path)
# write your code here
def clean_data(path):
    df = pd.read_csv(path)

    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'],format='%Y')
    df.fillna({'team':'No Team'},inplace=True)
    # example height entry: 6-9 / 2.06
    df['height'] = df['height'].str.split().str[-1].astype(float)
    # example weight entry: 250 lbs. / 113.4 kg.
    df['weight'] = df['weight'].str.split('/').str[-1].str.split().str[0].astype(float)
    # example salary entry: $12500000
    df['salary'] = df['salary'].str.replace('$', '').astype(float)
    # If outside the USA type as Not-USA
    df['country'] = df['country'].apply(lambda x: 'Not-USA' if x != 'USA' else 'USA')
    df['country'] = df['country'].apply(lambda x: 'Not-USA' if x != 'USA' else 'USA')
    df['draft_round'] = df['draft_round'].replace('Undrafted', "0")

    return df

"""df2 = clean_data(data_path)
print(df1.head())
print(df1['height'])
print(df2.head())"""