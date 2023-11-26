# Import libraries
import numpy as np
import data_clean as dclean
import feature_eng as feng
import data_split as dsplit
import data_standardization as dstd
import modelling as mdl

path = "account_receivable.csv"
df = dclean.get_df(path=path)
df = dclean.rename_columns(df)

# print(df.columns)

df = feng.feature_eng(df)

# Split dataset into Input-Output
X, y = dsplit.extract_input_output(data = df,
                          output_column_name = "is_on_time")


# Split input and output into Train-Test Dataset
X_train, X_test, y_train, y_test = dsplit.train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 12)




# asdfasdf
X_train_std, standardizer = dstd.get_standardizer(data = X_train)

# asdfasd
X_test_std = dstd.standardize(X_test, standardizer)


params = {
        'max_depth': np.arange(3, 11),
        'subsample': np.arange(0.5, 1.0, 0.1),
        'max_features': ['sqrt', 'log2', None]
        }

# asfdasdf
clf = mdl.train(X_train_std, y_train, params)

# asfasdfa
y_pred = mdl.predict(X_test_std, y_test, clf)

