import numpy as np
import rowmean  # предполагаем, что это ваш модуль после установки
import time

# Тестовые данные: большая матрица 1000x1000
matrix = np.random.rand(1000, 1000)  # Генерация случайной матрицы 1000x1000

# Эталонный результат с использованием NumPy
expected_means = np.mean(matrix, axis=1)

# Измерение времени выполнения функции NumPy
start_time_np = time.time()
expected_means = np.mean(matrix, axis=1)
end_time_np = time.time()
np_time = end_time_np - start_time_np

# Измерение времени выполнения вашей функции
start_time_rowmean = time.time()
calculated_means = rowmean.row_mean(matrix)
end_time_rowmean = time.time()
rowmean_time = end_time_rowmean - start_time_rowmean

# Вывод результатов
print("Expected means (first 10):", expected_means[:10])  # Показываем первые 10 значений для компактности
print("Calculated means (first 10):", calculated_means[:10])  # Показываем первые 10 значений для компактности
print(f"NumPy execution time: {np_time:.6f} seconds")
print(f"Rowmean execution time: {rowmean_time:.6f} seconds")

# Проверка, что результаты совпадают
if np.allclose(expected_means, calculated_means):
    print("Test passed.")
else:
    print("Test failed.")
