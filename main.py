import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import daal4py as d4p
from data_processing import *

X_train, X_test, y_train, y_test = download_amazon_sentences()
d2v_model = amazon_d2v_model()

# ----------------- PREDICTION WITH XGBOOST -----------------
n_estimators = 200
booster = 'gbtree'
max_depth = 50
max_leaves = 100

xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, n_jobs=4,  booster=booster,  max_depth=max_depth,
                              max_leaves=max_leaves, use_label_encoder=False,  eval_metric='mlogloss')
s = time.time()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print('-- XGBOOST --')
print('Time:', time.time() - s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", round(precision_score(y_test, y_pred), 5))
print("Recall:", round(recall_score(y_test, y_pred), 5))
print("F1 score:", round(f1_score(y_test, y_pred), 5))


# ------------- PREDICTION WITH INTEL'S XGBOOST -------------
nClasses = 5
featuresPerNode = 8
maxIterations = 500
minObservationsInLeafNode = 10

s = time.time()
daal_model = d4p.gbt_classification_training(nClasses=nClasses, maxIterations=maxIterations,
                                             minObservationsInLeafNode=minObservationsInLeafNode,
                                             featuresPerNode=featuresPerNode)
train_result = daal_model.compute(X_train, [y_train], weights=None)

daal_predict_algo = d4p.gbt_classification_prediction(
    nClasses=nClasses,
    nIterations=maxIterations,
    resultsToEvaluate="computeClassLabels|computeClassProbabilities")
daal_prediction = daal_predict_algo.compute(X_test, train_result.model)
daal_prediction = np.array([1 if int(daal_prediction.prediction[i][0]) else 0 for i in range(len(daal_prediction.prediction))])
acc = accuracy_score(y_test, daal_prediction)
print('-- DAAL4PY -- ')
print('Accuracy:', acc)
print('Time:', (time.time() - s) / 60, 'min')
