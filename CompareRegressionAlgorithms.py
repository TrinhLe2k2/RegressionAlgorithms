import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification

# Tạo dữ liệu giả định
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Hồi quy Tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 2. Hồi quy Logistic
logistic_model = LogisticRegression()
# Giảm giá trị của n_classes và n_clusters_per_class
X_classification, y_classification = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42, n_classes=2, n_clusters_per_class=1)
logistic_model.fit(X_classification, y_classification)

# 3. Hồi quy Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# 4. Hồi quy Đa thức
degree = 2
polyreg_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg_model.fit(X_train, y_train)

# 5. Hồi quy Từng bước
stepwise_model = LinearRegression()
stepwise_model.fit(X_train, y_train)


# 6. Hồi quy ElasticNet
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)

# Vẽ đồ thị
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, linear_model.predict(X_test), label='Linear Regression', linewidth=2)
plt.plot(X_test, logistic_model.predict(X_test), label='Logistic Regression', linewidth=2)
plt.plot(X_test, ridge_model.predict(X_test), label='Ridge Regression', linewidth=2)
plt.plot(X_test, polyreg_model.predict(X_test), label='Polynomial Regression', linewidth=2)
plt.plot(X_test, stepwise_model.predict(X_test), label='Stepwise Regression', linewidth=2)
plt.plot(X_test, elasticnet_model.predict(X_test), label='ElasticNet Regression', linewidth=2)

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Regression Algorithms Demo')
plt.show()
