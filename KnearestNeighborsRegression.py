# Import các thư viện cần thiết
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Diện tích nhà
y = 3 * X.squeeze() + np.random.randn(100) * 2  # Giá nhà (3 * diện tích + nhiễu)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình KNN
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn_model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Vẽ đồ thị để hiển thị dữ liệu và dự đoán
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.scatter(X_test, y_pred, color='red', label='Dự đoán')
plt.title('Hồi quy K-nearest neighbors: So sánh giá nhà thực tế và dự đoán')
plt.xlabel('Diện tích nhà')
plt.ylabel('Giá nhà')
plt.legend()
plt.show()
