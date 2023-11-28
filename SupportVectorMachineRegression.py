# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Tạo dữ liệu mẫu
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Diện tích nhà
y = 3 * X.squeeze() + np.random.randn(100) * 2  # Giá nhà (3 * diện tích + nhiễu)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình SVM hồi quy
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_svm = svm_model.predict(X_test)

# Đánh giá mô hình
mse_svm = mean_squared_error(y_test, y_pred_svm)
print(f'Mean Squared Error (SVM): {mse_svm}')

# Vẽ đồ thị để hiển thị dữ liệu và dự đoán
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.scatter(X_test, y_pred_svm, color='blue', label='Dự đoán (SVM)')
plt.title('Hồi quy Support Vector Machine: So sánh giá nhà thực tế và dự đoán')
plt.xlabel('Diện tích nhà')
plt.ylabel('Giá nhà')
plt.legend()
plt.show()
