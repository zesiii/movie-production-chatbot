
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



excel_path = 'ML/new_random_200_withsentiment.csv'
input_csv = 'ML/input_ML.csv'
# encodings_to_try = ['utf-8', 'ISO-8859-1']
# for encoding in encodings_to_try:
#     try:
#         df = pd.read_csv(excel_path, encoding='ISO-8859-1')
#         # inputdf = pd.read_csv(input_csv, encoding = encoding)
#         break
#     except UnicodeDecodeError:
#         print(f"Failed to read with encoding: {encoding}")

# if 'df' in locals():
#     print(df)
excel_path2 = 'ML/random_200_questions_revenue_sentiment(3).csv'
df = pd.read_csv(excel_path, encoding='ISO-8859-1')
inputdf = pd.read_csv(input_csv)
df_new_review = pd.read_csv(excel_path2)
df_structure = df_new_review.dtypes

def data_pipline(df, boxoffice, cossim):
    boxoffice = boxoffice[['Movie', 'revenue', 'budget']]
    df_total = df.merge(boxoffice, how='left', on='Movie')
    df_cleaned = df_total.dropna(subset=['revenue', 'gpt3.5_Prompted_sentiment'])
    df_cleaned['revenue'] = np.log(df_cleaned['revenue'])
    df_cleaned['budget'] = np.log(df_cleaned['budget'].astype(float))
    df_cleaned['gpt3.5_Prompted_score'] = (df_cleaned['gpt3.5_Prompted_score']
                                           .fillna(df_cleaned['gpt3.5_Prompted_score'].mean()))
    df_cleaned['Score'] = df_cleaned['Score'].fillna(df_cleaned['Score'].mean())
    df_cleaned = df_cleaned.sort_values(by='Movie')
    le = LabelEncoder()
    df_cleaned['Reviewer_encoded'] = le.fit_transform(df_cleaned['Reviewer'].astype(str))
    df_cleaned['Publish_encoded'] = le.fit_transform(df_cleaned['Publish'].astype(str))
    df_cleaned['gpt_sentiment'] = le.fit_transform(df_cleaned['gpt3.5_Prompted_sentiment'].astype(str))
    le.fit(df_cleaned['Movie'].unique())
    df_cleaned['Movie_encoded'] = le.transform(df_cleaned['Movie'].astype(str))
    cossim['Movie_encoded'] = le.transform(cossim['Movie'].astype(str))

    return df_cleaned, cossim

def fetch_input_train_test(df, movie_name):
    movie_list = list(pd.unique(df['Movie']))
    movie_test = movie_name
    movie_train = list(set(movie_list) - set(movie_test))
    finaldata_train = df[df['Movie'].isin(movie_train)]
    finaldata_test = df[df['Movie'].isin(movie_test)]
    return finaldata_train, finaldata_test
    # input_df = df[df['Movie'] == movie_name]
    # df_except_input = df[df['Movie'] != movie_name]
    # return finaldata_train, finaldata_test

# def train_test(df):
#     movie_list = list(pd.unique(df['Movie']))
#     random.seed(420)
#     movie_test = random.sample(movie_list, k=int(len(movie_list) * 0.2))
#     movie_train = list(set(movie_list) - set(movie_test))
#     finaldata_train = df[df['Movie'].isin(movie_train)]
#     finaldata_test = df[df['Movie'].isin(movie_test)]
#     return finaldata_train, finaldata_test

def positive_sentiment_ratio(df):
    positive_count = df[df['gpt3.5_Prompted_sentiment'] == 'Positive'].shape[0]
    negative_count = df[df['gpt3.5_Prompted_sentiment'] == 'Negative'].shape[0]
    total_count = df['gpt3.5_Prompted_sentiment'].count()
    positive_percentage = positive_count / total_count if total_count != 0 else 0
    negative_percentage = negative_count / total_count if total_count != 0 else 0

    return pd.Series({'positive_percentage': positive_percentage, 'negative_percentage': negative_percentage})


def data_preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['days_since_earliest'] = (df['Date'] - df['Date'].min()).dt.days
    # le = LabelEncoder()
    # df['Movie_encoded'] = le.fit_transform(df['Movie'].astype(str))
    # df['Reviewer_encoded'] = le.fit_transform(df['Reviewer'].astype(str))
    # df['Publish_encoded'] = le.fit_transform(df['Publish'].astype(str))
    # df['gpt_sentiment'] = le.fit_transform(df['gpt3.5_Prompted_sentiment'].astype(str))

    sentiment_ratios = df.groupby('Movie').apply(positive_sentiment_ratio)
    df = df.merge(sentiment_ratios, on='Movie', how='left')
    df = df.dropna()
    features_gpt = [
        'Score', 'days_since_earliest', 'Movie_encoded',
        'Reviewer_encoded', 'Publish_encoded', 'gpt_sentiment',
        'gpt3.5_Prompted_score', 'positive_percentage', 'negative_percentage',
        'budget']
    df_x = df[features_gpt]
    df_y = df['revenue']
    return df_x, df_y


def train_evaluate_network(optimizer, neurons, epochs, batch_size, train_x, train_y, test_x, test_y):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(train_x.shape[1],)),
        Dense(neurons, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
    predictions = model.predict(test_x).flatten()
    return predictions

def weighted_output(prediction, MLdata_test_x, inputdf):
    inputdf['normalized_cosscore'] = inputdf['cosscore']/inputdf['cosscore'].sum()
    # print(f"inputdf normalized{inputdf}")
    df = MLdata_test_x.merge(inputdf, how= 'left', on ='Movie_encoded')
    df['prediction'] = prediction
    df['exp_prediction'] = np.exp(df['prediction'])
    grouped = df.groupby('Movie_encoded')
    total_sum = 0
    for category, group in grouped:
        weighted_sum = (group['exp_prediction'] * group['normalized_cosscore']).sum()
        a = weighted_sum / len(group)
        total_sum += a
    # print('weighted_output', total_sum)


    return total_sum

    # pass

def main():
    finaldata, cossim = data_pipline(df, df_new_review, inputdf)
    finaldata_train, finaldata_test = fetch_input_train_test(finaldata, inputdf['Movie'])
    # finaldata_train, finaldata_test = train_test(df_except_input)
    MLdata_train_x, MLdata_train_y = data_preprocess(finaldata_train)
    MLdata_test_x, MLdata_test_y = data_preprocess(finaldata_test)
    # print(MLdata_test_x['Movie_encoded'])
    # print('======================')
    # print(MLdata_test_y)
    # print('======================')
    # input_df_x, input_df_y = data_preprocess(input_df)
    # Normalize data for NN
    scaler = StandardScaler()
    MLdata_train_x_scaled = scaler.fit_transform(MLdata_train_x)
    MLdata_test_x_scaled = scaler.transform(MLdata_test_x)
    # MLdata_input_x = scaler.transform(input_df_x)
    # CNN
    nn_best_params = {
        'epochs': 10,
        'batch_size': 100,
        'optimizer': 'rmsprop',
        'neurons': 64
    }
    nn_predictions = train_evaluate_network(nn_best_params['optimizer'], nn_best_params['neurons'],
                                            nn_best_params['epochs'], nn_best_params['batch_size'],
                                            MLdata_train_x_scaled, MLdata_train_y, MLdata_test_x_scaled, MLdata_test_y)

    # other models
    models = {
        'Random Forest': RandomForestRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=200,
                                               random_state=42),
        # 'Linear Regression': LinearRegression(),
        # 'Ridge Regression': Ridge(alpha=0.0001),
        # 'Lasso Regression': Lasso(alpha=0.0001),
        'Support Vector Regression': SVR(C=0.05, gamma='scale'),
        'Gradient Boosting Regressor': GradientBoostingRegressor(learning_rate=0.001, max_depth=4, n_estimators=200)
    }
    for model in models.values():
        model.fit(MLdata_train_x, MLdata_train_y)
    predictions = [model.predict(MLdata_test_x) for model in models.values()]

    # bagging
    combined_predictions = np.mean([*predictions, nn_predictions], axis=0)
    # result
    bagged_rmse = mean_squared_error(MLdata_test_y, combined_predictions, squared=False)

    #print(bagged_rmse)
    #print('======================')

    # get weighted result
    output = weighted_output(combined_predictions, MLdata_test_x, inputdf)
    print(output)

    return output
    #print(output[['Movie_encoded', 'prediction', 'exp_prediction', 'normalized_cosscore']].tail())

if __name__ == "__main__":
    main()
    # print(inputdf['Movie'])
