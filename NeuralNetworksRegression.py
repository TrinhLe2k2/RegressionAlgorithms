import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Diện tích nhà
y = 3 * X.squeeze() + np.random.randn(100) * 2  # Giá nhà (3 * diện tích + nhiễu)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Các thông số của Mạng Nơ-ron
kich_thuoc_nhap = 1
kich_thuoc_an = 64
kich_thuoc_ra = 1
tasa_hoc = 0.01
so_lan_lap = 100

# Khởi tạo trọng số và độ lệch
trong_so_nhap_an = np.random.randn(kich_thuoc_nhap, kich_thuoc_an)
do_lech_nhap_an = np.zeros((1, kich_thuoc_an))
trong_so_an_ra = np.random.randn(kich_thuoc_an, kich_thuoc_ra)
do_lech_an_ra = np.zeros((1, kich_thuoc_ra))

# Huấn luyện Mạng Nơ-ron
for epoch in range(so_lan_lap):
    # Lan truyền tiến
    dau_vao_laymanh = np.dot(X_train, trong_so_nhap_an) + do_lech_nhap_an
    lop_an = np.maximum(0, dau_vao_laymanh)  # Hàm kích hoạt ReLU
    dau_ra = np.dot(lop_an, trong_so_an_ra) + do_lech_an_ra

    # Tính mất mát (Mean Squared Error)
    mat_mat = np.mean((dau_ra - y_train.reshape(-1, 1))**2)

    # Lan truyền ngược
    loi_ra = dau_ra - y_train.reshape(-1, 1)
    loi_an = np.dot(loi_ra, trong_so_an_ra.T)

    # Cập nhật trọng số và độ lệch bằng gradient descent
    trong_so_an_ra -= tasa_hoc * np.dot(lop_an.T, loi_ra) / len(X_train)
    do_lech_an_ra -= tasa_hoc * np.sum(loi_ra, axis=0, keepdims=True) / len(X_train)
    trong_so_nhap_an -= tasa_hoc * np.dot(X_train.T, loi_an * (lop_an > 0)) / len(X_train)
    do_lech_nhap_an -= tasa_hoc * np.sum(loi_an * (lop_an > 0), axis=0, keepdims=True) / len(X_train)

    # In mất mát sau mỗi 10 lần lặp
    if epoch % 10 == 0:
        print(f'Lần lặp {epoch}, Mất mát: {mat_mat}')

# Dự đoán trên tập kiểm tra
dau_vao_laymanh_test = np.dot(X_test, trong_so_nhap_an) + do_lech_nhap_an
lop_an_test = np.maximum(0, dau_vao_laymanh_test)
du_doan = np.dot(lop_an_test, trong_so_an_ra) + do_lech_an_ra

# Đánh giá mô hình
mse = mean_squared_error(y_test, du_doan.squeeze())
print(f'Mean Squared Error (Mạng Nơ-ron): {mse}')

# Vẽ đồ thị để hiển thị dữ liệu và dự đoán
plt.scatter(X_test, y_test, color='black', label='Thực tế')
plt.scatter(X_test, du_doan, color='blue', label='Dự đoán (Mạng Nơ-ron)')
plt.title('Hồi quy Mạng Nơ-ron: So sánh giá nhà thực tế và dự đoán')
plt.xlabel('Diện tích nhà (chuẩn hóa)')
plt.ylabel('Giá nhà')
plt.legend()
plt.show()
