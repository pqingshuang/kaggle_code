import os
import tensorflow as tf
import itertools

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
# matplotlib.use('TkAgg')

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')

selected_features = ['penalty','time', 'n_samples', 'max_iter',
                         'n_features', 'n_classes', 'flip_y', 'n_informative',
                         'n_jobs','n_clusters_per_class']

test_selected_feratures= ['penalty', 'n_samples', 'max_iter',
                         'n_features', 'n_classes', 'flip_y', 'n_informative',
                         'n_jobs','n_clusters_per_class']

train.drop('id',axis = 1, inplace = True)
train = train[selected_features]
train_numerical = train.select_dtypes(exclude=['object'])
train_numerical.fillna(0,inplace = True)
train_categoric = train.select_dtypes(include=['object'])
train_categoric.fillna('NONE',inplace = True)
train = train_numerical.merge(train_categoric, left_index = True, right_index = True)

test = pd.read_csv('test.csv')
ID = test.id
test.drop('id',axis = 1, inplace = True)
test = test[test_selected_feratures]



test_numerical = test.select_dtypes(exclude=['object'])
test_numerical.fillna(0,inplace = True)
test_categoric = test.select_dtypes(include=['object'])
test_categoric.fillna('NONE',inplace = True)
test = test_numerical.merge(test_categoric, left_index = True, right_index = True)


# outlines
from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train_numerical)
y_noano = clf.predict(train_numerical)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
# print(y_noano[y_noano['Top'] == 1].index.values)
#
train_numerical = train_numerical.iloc[y_noano[y_noano['Top'] == 1].index.values]
train_numerical.reset_index(drop = True, inplace = True)
#
train_categoric = train_categoric.iloc[y_noano[y_noano['Top'] == 1].index.values]
train_categoric.reset_index(drop = True, inplace = True)
#
train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)

col_train_num = list(train_numerical.columns)
col_train_num_bis = list(train_numerical.columns)
#
col_train_cat = list(train_categoric.columns)

col_train_num_bis.remove('time')

mat_train = np.matrix(train_numerical)
mat_test  = np.matrix(test_numerical)
mat_new = np.matrix(train_numerical.drop('time',axis = 1))
mat_y = np.array(train.time)

# print(mat_train)
#
prepro_y = MinMaxScaler()
prepro_y.fit(mat_y.reshape(360,1))
#
prepro = MinMaxScaler()
prepro.fit(mat_train)
#
prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

# train_num_scale = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
# test_num_scale  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)
train_num_scale = pd.DataFrame(prepro.transform(mat_train),columns = col_train_num)
test_num_scale  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_num_bis)
#
train[col_train_num] = pd.DataFrame(prepro.transform(mat_train),columns = col_train_num)
test[col_train_num_bis]  = test_num_scale
#
# List of features
COLUMNS = col_train_num
FEATURES = col_train_num_bis
LABEL = "time"
#
FEATURES_CAT = col_train_cat
#
engineered_features = []
#
for continuous_feature in FEATURES:
    engineered_features.append(
        tf.contrib.layers.real_valued_column(continuous_feature))

for categorical_feature in FEATURES_CAT:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        categorical_feature, hash_bucket_size=1000)

    engineered_features.append(
        tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16, combiner="sum"))

# Training set and Prediction set with the features to predict
training_set = train[FEATURES + FEATURES_CAT]
prediction_set = train.time
#
# Train and Test
# x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES + FEATURES_CAT],
#                                                     prediction_set, test_size=0.33, random_state=42)
# y_train = pd.DataFrame(y_train, columns=[LABEL])

# training_set = pd.DataFrame(x_train, columns=FEATURES + FEATURES_CAT).merge(y_train, left_index=True, right_index=True)
"" \
"" \
"" \
"Test"
train_X = training_set[FEATURES + FEATURES_CAT]
train_y = pd.DataFrame(prediction_set, columns=[LABEL])
testing_sub = test[FEATURES + FEATURES_CAT]
training_set = train_X.merge(train_y, left_index=True, right_index=True)
# Training for submission
# training_sub = training_set[FEATURES + FEATURES_CAT]
# testing_sub = test[FEATURES + FEATURES_CAT]
# #
# y_test = pd.DataFrame(y_test, columns = [LABEL])
# testing_set = pd.DataFrame(x_test, columns = FEATURES + FEATURES_CAT).merge(y_test, left_index = True, right_index = True)
#
# training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)
# testing_set[FEATURES_CAT] = testing_set[FEATURES_CAT].applymap(str)

#
def input_fn_new(data_set, training=True):
    continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(data_set[k].size)], values=data_set[k].values, dense_shape=[data_set[k].size, 1])
    for k in FEATURES_CAT}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))

    if training == True:
        # Converts the label column into a constant Tensor.
        label = tf.constant(data_set[LABEL].values)

        # Returns the feature columns and the label.
        return feature_cols, label

    return feature_cols

# "" \
# "" \
# "" \
"Model"
regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])
categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(training_set[k].size)], values = training_set[k].values, dense_shape = [training_set[k].size, 1]) for k in FEATURES_CAT}

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn = lambda: input_fn_new(training_set) , steps=2000)
# ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training = True), steps=1)
# loss_score4 = ev["loss"]
# print("Final Loss on the testing set: {0:f}".format(loss_score4))

y_predict = regressor.predict(input_fn=lambda: input_fn_new(testing_sub, training = False))
# print(y_predict)
def to_submit(pred_y,name_out):
    y_predict = list(itertools.islice(pred_y, test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns = ['time'])
    # y_predict = y_predict.join(ID)
    decimals = pd.Series([2], index=['time'])
    y_predict = y_predict.round(decimals)
    y_predict.to_csv(name_out + '.csv',index=False)
#
#
to_submit(y_predict, "submission_cont_categ1102-2")