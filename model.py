import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from sklearn.model_selection import train_test_split

data = pd.read_csv('feature_set_dem.csv')
X = np.array([data['prp_count'], data['VP_count'], data['NP_count'], #data['DT_count'], 
                data['prp_noun_ratio'], data['word_sentence_ratio'],
                data['count_pauses'], data['count_unintelligible'], 
                data['count_repetitions'], data['ttr'], data['R'],
                data['ARI'], data['CLI']])

X = X.T

Y = data['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=212)
train_samples, n_features = X.shape

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

joblib.dump(lda, 'lda_dementia_model.pkl')