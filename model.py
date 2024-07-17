import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

data = pd.read_csv('feature_set_dem.csv')

X = data[['prp_count', 'VP_count', 'NP_count', 'prp_noun_ratio', 'word_sentence_ratio',
          'count_pauses', 'count_unintelligible', 'count_repetitions', 'ttr', 'R', 'ARI', 'CLI']].values
Y_dementia = data['Category'].values
Y_emergency = data['Emergency'].values

lda_dementia = LinearDiscriminantAnalysis()
lda_dementia.fit(X, Y_dementia)
joblib.dump(lda_dementia, 'lda_dementia_model.pkl')

lda_emergency = LinearDiscriminantAnalysis()
lda_emergency.fit(X, Y_emergency)
joblib.dump(lda_emergency, 'lda_emergency_model.pkl')
