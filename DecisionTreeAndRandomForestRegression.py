# Import các thư viện cần thiết
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Diện tích nhà
y = 3 * X.squeeze() + np.random.randn(100) * 2  # Giá nhà (3 * diện tích + nhiễu)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Xây dựng mô hình Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Đánh giá mô hình Decision Tree
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Mean Squared Error (Decision Tree): {mse_dt}')

# Đánh giá mô hình Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Mean Squared Error (Random Forest): {mse_rf}')

# Vẽ đồ thị để hiển thị dữ liệu và dự đoán
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.scatter(X_test, y_pred_dt, color='blue', label='Dự đoán (Decision Tree)')
plt.scatter(X_test, y_pred_rf, color='red', label='Dự đoán (Random Forest)')
plt.title('Hồi quy Decision Tree và Random Forest: So sánh giá nhà thực tế và dự đoán')
plt.xlabel('Diện tích nhà')
plt.ylabel('Giá nhà')
plt.legend()
plt.show()
