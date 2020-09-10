import numpy as np
import pandas as pd
import os
import featuretools as ft
import utils as utils

# data (csv) --> entityset (tables) --> feature_matrix (dataframe) --> feature_matrix_encoded (dataframe) --> features.csv
# label (csv) --> label_df (dataframe) --> labels (series) --> labels.csv
# --> tpot.fit(features.csv, labels.csv)


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



feature_matrix_encoded, features_encoded = ft.encode_features(feature_matrix, features)




np.savetxt("features.csv", feature_matrix_encoded.values, delimiter=",")
np.savetxt("labels.csv", y_binary, delimiter=",")





feature_matrix.to_csv('output/feature_matrix.csv', encoding='utf-8')
feature_matrix_encoded.to_csv('output/feature_matrix_encoded.csv', encoding='utf-8')

features_df = pd.DataFrame(features)
features_df.to_csv('output/features.csv', encoding='utf-8')






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



