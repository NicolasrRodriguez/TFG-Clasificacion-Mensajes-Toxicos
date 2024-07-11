
import conjuntosEVT
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


scaler = StandardScaler()
X_train = scaler.fit_transform(conjuntosEVT.train_f)

clf = LogisticRegression(max_iter=1000, random_state=42)

cv = StratifiedKFold(5)

rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='accuracy', min_features_to_select=1, n_jobs=-1)

rfecv.fit(X_train, conjuntosEVT.train_c)

print(f"Número óptimo de características : {rfecv.n_features_}")



cv_results = pd.DataFrame(rfecv.cv_results_)
print(cv_results.describe())
print(cv_results.columns)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.plot(range(1, len(cv_results["mean_test_score"]) + 1), cv_results["mean_test_score"])
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()