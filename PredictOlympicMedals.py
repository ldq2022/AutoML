import numpy as np
import pandas as pd
import os


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

import featuretools as ft

import utils as utils





DATA_DIR = os.path.join(os.getcwd(),"data/olympic_games_data")
es = utils.load_entityset(data_dir=DATA_DIR)

# Load a pre-made labels table for supervised learning
label_file = os.path.join(DATA_DIR, "num_medals_by_country_labels.csv")
label_df = pd.read_csv(label_file,
                       parse_dates=['Olympics Date'],
                       encoding='utf-8',
                       usecols=['Number of Medals', 'Olympics Date', 'Country'])
label_df.sort_values(['Olympics Date', 'Country'], inplace=True)

dates = label_df['Olympics Date']
labels = label_df['Number of Medals']
y_binary = (labels >= 10).values



cutoff_times = label_df[['Country', 'Olympics Date']].rename(columns={'Country': 'Code', 'Olympics Date': 'time'})
cutoff_times.tail()



agg_primitives = ['Sum', 'Std', 'Max', 'Min', 'Mean',
                  'Count', 'Percent_True', 'Num_Unique',
                  'Mode', 'Trend', 'Skew']

feature_matrix, features = ft.dfs(
    entityset=es,
    target_entity="countries",
    trans_primitives=[],
    agg_primitives=agg_primitives,
    max_depth=3,
    cutoff_time=cutoff_times,
    verbose=True
)

print("{} features generated".format(len(features)))



features[-10:]


feature_matrix_encoded, features_encoded = ft.encode_features(feature_matrix, features)

pipeline_preprocessing = [("imputer",
                           SimpleImputer()),
                          ("scaler", RobustScaler(with_centering=True))]
feature_matrix_encoded.tail()


splitter = utils.TimeSeriesSplitByDate(dates=dates, earliest_date=pd.Timestamp('1/1/1960'))
X = feature_matrix_encoded.values

rf_clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
pipeline = Pipeline(pipeline_preprocessing + [('rf_clf', rf_clf)])
binary_scores = utils.fit_and_score(X, y_binary, splitter, pipeline, _type='classification')
"Average AUC score is {} with standard dev {}".format(
        round(binary_scores['roc_auc'].mean(), 3),
        round(np.std(binary_scores['roc_auc']), 3)
)


binary_scores.set_index('Olympics Year')['roc_auc'].plot(title='AUC vs. Olympics Year')

split, year = 5, '1984'
train, test = splitter.split(X, y_binary)[split]
# pipeline.fit(X[train], y_binary[train])
x_train = X[train]
y_train = y_binary[train]


np.savetxt("x_train.csv", x_train, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
# y_pred = pipeline.predict(X[test])
# cm = confusion_matrix(y_binary[test], y_pred)
# utils.plot_confusion_matrix(cm, ['Won < 10 Medals', 'Won >= 10 Medals'], title=year)
#
#
# # 2004 = 10th split
# split, year = 10, '2004'
# train, test = splitter.split(X, y_binary)[split]
# pipeline.fit(X[train], y_binary[train])
# y_pred = pipeline.predict(X[test])
# cm = confusion_matrix(y_binary[test], y_pred)
# utils.plot_confusion_matrix(cm, ['Won < 10 Medals', 'Won >= 10 Medals'], title=year)
#
#
#
# # Get feature importances for every year
# feature_imp = utils.get_feature_importances(pipeline,
#                                             feature_matrix_encoded,
#                                             (labels >= 10), splitter)
#
# # Show 10 most important features for 1984
# test_date = pd.Timestamp('6/29/1984')
#
#
# # Save output files
#
# import os
#
# try:
#     os.mkdir("output")
# except:
#     pass
#
# feature_matrix_encoded.to_csv('output/feature_matrix_encoded.csv', encoding='utf-8')
# cutoff_times.to_csv('output/cutoff_times.csv', encoding='utf-8')

print("=== Train and Test Set has been created ===")



