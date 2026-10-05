import matplotlib.pyplot as plt
from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_histograms(matrix, count=3):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))   # два графика рядом
    for i in range(count):
        left.hist(matrix[i], bins=20, alpha=0.5, label=f"строка {i}")
        right.hist(matrix[:, i], bins=20, alpha=0.5, label=f"столбец {i}")
    left.set_title("Строки")
    left.legend()
    right.set_title("Столбцы")
    right.legend()
    plt.show()


stats = matrix_statistics(RandomMatrixInput(rows=500, columns=500, mean=0, std=1, seed=42))
plot_histograms(stats.matrix)