# Import thư viện
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

# Khởi tạo mô hình
model = LinearRegression()

# Fit mô hình
model.fit(X, Y)

# Dự đoán
Y_pred = model.predict(X)

# Kết Quả Dự đoán
print("Dự đoán:", Y_pred)

# Hiển thị biểu đồ
plt.scatter(X, Y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, Y_pred, color='red', label='Dự đoán', linewidth=2)
plt.title('Hồi Quy Tuyến Tính: Giá nhà dựa trên diện tích')
plt.xlabel('Diện tích nhà')
plt.ylabel('Giá nhà')
plt.legend()
plt.show()
