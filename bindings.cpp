#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::vector<double> row_mean(const std::vector<std::vector<double>>& matrix) {
    std::vector<double> means;
    for (const auto& row : matrix) {
        double sum = 0;
        for (double val : row) {
            sum += val;
        }
        means.push_back(sum / row.size());
    }
    return means; // Возвращаем вектор средних значений
}

PYBIND11_MODULE(rowmean, m) {
    m.def("row_mean", &row_mean, "Вычисление среднего значения строк матрицы");
}
