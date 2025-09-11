import pandas as pd
import numpy as np
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

def feature_data(df):

    # Series object which has 'versions' entries in 2000 + %y format
    version_to_time = pd.Series(df['version'].str.extract('(\d\d$)').iloc[:, 0].astype(int) + 2000)

    # Change unique values in a version column to datetime object
    unique_values = pd.to_datetime(version_to_time.unique(), format='%Y')

    # Engineer the age feature by subtracting b_day column from version. Calculate the value as year;
    df['age'] = np.ceil((pd.to_datetime(version_to_time, format='%Y') - pd.to_datetime(df['b_day'])).dt.days / 365.25)

    # Engineer the experience feature by subtracting draft_year column from version. Calculate the value as year;
    df['experience'] = np.ceil((pd.to_datetime(version_to_time, format='%Y') - pd.to_datetime(df['draft_year'])).dt.days / 365.25)

    # Engineer bmi feature representing the bmi of a player
    df['bmi'] = df.weight / np.pow(df.height, 2)

    # Drop 'version', 'b_day', 'draft_year', 'weight', 'height' columns
    df = df.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], axis=1)

    # Drop high cardinality features (>50 unique values)
    high_cardinality_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 50]
    df = df.drop(columns=high_cardinality_cols)

    return df

if __name__ == "__main__":
    df_cleaned = clean_data(data_path)
    df = feature_data(df_cleaned)
    print(df[['age', 'experience', 'bmi']].head())
