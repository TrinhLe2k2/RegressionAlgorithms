# Import thư viện
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([0, 0, 1, 1, 1])

# Khởi tạo mô hình
model = LogisticRegression()

# Fit mô hình
model.fit(X, Y)

# Dự đoán xác suất
proba = model.predict_proba(X)[:, 1]

# Kết Quả Dự đoán
print("Dự đoán:", proba)

# Hiển thị biểu đồ
plt.scatter(X, Y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, proba, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Dự đoán')
plt.title('Hồi Quy Logistic: Phân loại email là spam')
plt.xlabel('Số từ khóa')
plt.ylabel('Xác suất là spam')
plt.legend()
plt.show()
