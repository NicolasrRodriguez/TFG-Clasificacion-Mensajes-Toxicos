
import conjuntosEVT
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train = scaler.fit_transform(conjuntosEVT.train_f)

clf = LogisticRegression(max_iter=1000, random_state=42)

selector = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy', n_jobs=-1)

selector.fit(X_train, conjuntosEVT.train_c)

print(f"Número óptimo de características : {selector.n_features_}")

cv_results = pd.DataFrame(selector.cv_results_)
print(cv_results.describe())


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=selector.n_features_,#no se que se pone aquí@@@@
    y=cv_results['mean_test_score'],
    yerr=cv_results['std_test_score'],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()
