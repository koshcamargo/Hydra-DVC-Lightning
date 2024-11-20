#include <vector>

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
