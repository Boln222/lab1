"""Заготовки задач на NumPy."""
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    sum = 0
    for i in range(len(matrices)):
        sum += matrices[i] @ vectors[i]
    return sum


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    rows, cols = matrix.shape
    result = np.zeros((rows,cols), dtype = int)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] > threshold:
                result[i][j] = 1
    return result 


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for rows in matrix:
        result.append(np.unique(rows).tolist())
    return result

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for column in matrix.T:
        result.append(np.unique(column).tolist())
    return result

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)                          # генератор с зерном
    matrix = rng.normal(mean, std, size=(rows, columns))       # случайная матрица

    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),         # среднее каждой строки
        column_means=matrix.mean(axis=0),      # среднее каждого столбца
        row_variances=matrix.var(axis=1),      # дисперсия каждой строки
        column_variances=matrix.var(axis=0),   # дисперсия каждого столбца
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows,columns))
    return np.where((i+j) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color)

    top = (image_height - height) // 2
    left = (image_width - width) // 2
    
    image[top:top + height, left:left + width] = shape_color
    return image

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color)
    y0 = (image_height - 1) / 2
    x0 = (image_width - 1) / 2
    y, x = np.indices((image_height, image_width))
    inside = (x - x0) ** 2 / semi_axis_x ** 2 + (y - y0) ** 2 / semi_axis_y ** 2 <= 1
    image[inside] = shape_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.array(values, dtype=float)

    mean = values.mean()
    variance = values.var()
    std = values.std()

    left = values[:-2]
    middle = values[1:-1]
    right = values[2:]

    maxima = np.where((middle > left) & (middle > right))[0] + 1
    minima = np.where((middle < left) & (middle < right))[0] + 1

    windows = sliding_window_view(values, window)
    moving_average = windows.mean(axis=1)

    return TimeSeriesStatistics(
        mean=float(mean),
        variance=float(variance),
        std=float(std),
        local_maxima_indices=maxima.tolist(),
        local_minima_indices=minima.tolist(),
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.array(labels, dtype=int)

    if class_count is None:
        class_count = labels.max() + 1

    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result
