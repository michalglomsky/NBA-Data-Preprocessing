import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
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

def multicol_data(df):

    # Take just numerical columns
    df_numerical = df.select_dtypes(include='number')

    # Create a correlation matrix
    corr_matrix = df_numerical.corr()

    # Create a matrix with salary correlations
    corr_salary = corr_matrix['salary'].abs()

    # Drop salary from the feature matrix
    corr_matrix = corr_matrix.drop('salary',axis=0).drop('salary',axis=1)

    # A list of columns to drop
    columns_to_drop = set()

    # Iterate through the feature matrix and analyze the correlations
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # Check if the correlation is above the threshold (using absolute value)
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]

                # Compare the two features' correlations with the salary
                if corr_salary[feature1] < corr_salary[feature2]:
                    columns_to_drop.add(feature1)
                else:
                    columns_to_drop.add(feature2)

    # Drop the chosen feature columns
    df = df.drop(columns=columns_to_drop)
    return df

def transform_data(df):

    # Seperate numerical and categorical features
    num_feat_df = df.select_dtypes('number')
    cat_feat_df = df.select_dtypes('object')

    # Seperating salary column from features
    y = num_feat_df['salary']
    num_feat_df = num_feat_df.drop('salary',axis=1)

    # Standardize numerical features
    scaler = StandardScaler()
    scaled_num_df = pd.DataFrame(
        scaler.fit_transform(num_feat_df),
        columns=num_feat_df.columns,
        index=num_feat_df.index
    )

    # Ordinal encode categorical features
    onehot = OneHotEncoder(sparse_output=False)
    cat_feat_df_transformed = onehot.fit_transform(cat_feat_df)
    og_categories = onehot.categories_
    list_of_columns = np.concatenate(og_categories).ravel().tolist()
    encoded_cat_df = pd.DataFrame(
        cat_feat_df_transformed,
        columns=list_of_columns,
        index=cat_feat_df.index
    )

    # Merge numerical and categorical features
    X = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

    return X, y

if __name__ == "__main__":
    df_cleaned = clean_data(data_path)
    df_featured = feature_data(df_cleaned)
    df = multicol_data(df_featured)

    X, y = transform_data(df)

    answer = {
        'shape': [X.shape, y.shape],
        'features': list(X.columns),
    }
    print(answer)
