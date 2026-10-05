import pytest
import numpy as np
from grader_contracts.numpy_tasks import MatrixVectorBatchInput, BinarizeInput,  MatrixInput, ChessInput, RectangleInput, EllipseInput, TimeSeriesInput, OneHotInput
from numpy_tasks import sum_prod, binarize, unique_rows, unique_columns, chess, draw_rectangle, draw_ellipse, analyze_time_series, one_hot


def test_sum_prod_example():
    # 2 пары, матрицы 2×2
    matrices = [np.array([[1, 2], [3, 4]]),
                np.array([[1, 0], [0, 1]])]
    vectors = [np.array([[5], [6]]),
               np.array([[1], [1]])]
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert np.array_equal(result, np.array([[18], [40]]))
    assert result.shape == (2, 1)


def test_sum_prod_single_pair():
    # p = 1: всего одна пара, складывать не с чем
    matrices = [np.array([[2, 0], [0, 3]])]
    vectors = [np.array([[4], [5]])]
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert np.array_equal(result, np.array([[8], [15]]))


def test_sum_prod_identity():
    # единичные матрицы не меняют вектор, ответ = просто сумма векторов
    eye = np.array([[1, 0], [0, 1]])
    matrices = [eye, eye]
    vectors = [np.array([[1], [2]]),
               np.array([[3], [4]])]
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert np.array_equal(result, np.array([[4], [6]]))


def test_sum_prod_zero_vectors():
    # нулевые векторы дают ноль, какой бы ни была матрица
    matrices = [np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]])]
    vectors = [np.array([[0], [0]]),
               np.array([[0], [0]])]
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert np.array_equal(result, np.array([[0], [0]]))


def test_sum_prod_size_one():
    # n = 1: матрицы 1×1, по сути просто числа: 2·4 + 3·5 = 23
    matrices = [np.array([[2]]), np.array([[3]])]
    vectors = [np.array([[4]]), np.array([[5]])]
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert np.array_equal(result, np.array([[23]]))
    assert result.shape == (1, 1)

def test_binarize_basic():
    # 3 равно порогу — это НЕ «строго больше», поэтому 0
    m = np.array([[1, 5], [3, 7]])
    result = binarize(BinarizeInput(matrix=m, threshold=3))
    assert np.array_equal(result, np.array([[0, 1], [0, 1]]))


def test_binarize_not_square():
    m = np.array([[1, 5, 9], [3, 7, 2]])
    result = binarize(BinarizeInput(matrix=m, threshold=3))
    assert np.array_equal(result, np.array([[0, 1, 1], [0, 1, 0]]))


def test_binarize_does_not_change_original():
    m = np.array([[1, 5], [3, 7]])
    binarize(BinarizeInput(matrix=m, threshold=3))
    assert np.array_equal(m, np.array([[1, 5], [3, 7]]))


def test_binarize_all_below():
    m = np.array([[1, 2], [3, 4]])
    result = binarize(BinarizeInput(matrix=m, threshold=10))
    assert np.array_equal(result, np.array([[0, 0], [0, 0]]))

# ---------- unique_rows ----------

def test_unique_rows_example():
    m = np.array([[1, 2, 2],
                  [3, 3, 3],
                  [1, 4, 1]])
    assert unique_rows(MatrixInput(matrix=m)) == [[1, 2], [3], [1, 4]]


def test_unique_rows_all_same():
    # все элементы одинаковые — в каждой строке остаётся одно значение
    m = np.array([[5, 5],
                  [5, 5]])
    assert unique_rows(MatrixInput(matrix=m)) == [[5], [5]]


def test_unique_rows_all_different():
    # повторов нет — строки остаются как есть
    m = np.array([[1, 2],
                  [3, 4]])
    assert unique_rows(MatrixInput(matrix=m)) == [[1, 2], [3, 4]]


def test_unique_rows_sorted():
    # np.unique сортирует значения
    m = np.array([[3, 1, 2, 1]])
    assert unique_rows(MatrixInput(matrix=m)) == [[1, 2, 3]]


def test_unique_rows_not_square():
    m = np.array([[1, 1, 2],
                  [3, 4, 3]])
    assert unique_rows(MatrixInput(matrix=m)) == [[1, 2], [3, 4]]


def test_unique_rows_negative_and_float():
    m = np.array([[-1.5, 2.0, -1.5]])
    assert unique_rows(MatrixInput(matrix=m)) == [[-1.5, 2.0]]


# ---------- unique_columns ----------

def test_unique_columns_example():
    m = np.array([[1, 2, 2],
                  [3, 3, 3],
                  [1, 4, 1]])
    assert unique_columns(MatrixInput(matrix=m)) == [[1, 3], [2, 3, 4], [1, 2, 3]]


def test_unique_columns_all_same():
    m = np.array([[5, 5],
                  [5, 5]])
    assert unique_columns(MatrixInput(matrix=m)) == [[5], [5]]


def test_unique_columns_all_different():
    m = np.array([[1, 2],
                  [3, 4]])
    assert unique_columns(MatrixInput(matrix=m)) == [[1, 3], [2, 4]]


def test_unique_columns_not_square():
    # 2 строки, 3 столбца — в ответе 3 списка
    m = np.array([[1, 1, 2],
                  [3, 4, 3]])
    assert unique_columns(MatrixInput(matrix=m)) == [[1, 3], [1, 4], [2, 3]]


def test_unique_columns_single_column():
    # один столбец — в ответе один список
    m = np.array([[1],
                  [1],
                  [2]])
    assert unique_columns(MatrixInput(matrix=m)) == [[1, 2]]

def test_chess_example():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    assert np.array_equal(result, np.array([[0, 1, 0, 1],
                                            [1, 0, 1, 0],
                                            [0, 1, 0, 1]]))


def test_chess_one_cell():
    # 1×1 — только левый верхний элемент, он равен a
    result = chess(ChessInput(rows=1, columns=1, first=5, second=7))
    assert np.array_equal(result, np.array([[5]]))


def test_chess_one_row():
    result = chess(ChessInput(rows=1, columns=5, first=5, second=7))
    assert np.array_equal(result, np.array([[5, 7, 5, 7, 5]]))


def test_chess_one_column():
    result = chess(ChessInput(rows=4, columns=1, first=5, second=7))
    assert np.array_equal(result, np.array([[5], [7], [5], [7]]))


def test_chess_other_numbers():
    # не только 0 и 1: отрицательные и дробные
    result = chess(ChessInput(rows=2, columns=2, first=-1, second=2.5))
    assert np.array_equal(result, np.array([[-1, 2.5], [2.5, -1]]))

RED = (255, 0, 0)
BLACK = (0, 0, 0)


# ---------- draw_rectangle ----------

def make_rectangle():
    # картинка 4×6, прямоугольник 2×2 → строки 1–2, столбцы 2–3
    return draw_rectangle(RectangleInput(width=2, height=2,
                                         image_height=4, image_width=6,
                                         shape_color=RED, background_color=BLACK))


def test_rectangle_shape():
    assert make_rectangle().shape == (4, 6, 3)


def test_rectangle_inside():
    image = make_rectangle()
    assert np.array_equal(image[1, 2], RED)   # левый верхний угол прямоугольника
    assert np.array_equal(image[2, 3], RED)   # правый нижний угол прямоугольника


def test_rectangle_outside():
    image = make_rectangle()
    assert np.array_equal(image[0, 2], BLACK)  # над прямоугольником
    assert np.array_equal(image[3, 2], BLACK)  # под ним
    assert np.array_equal(image[1, 1], BLACK)  # слева
    assert np.array_equal(image[1, 4], BLACK)  # справа


def test_rectangle_whole_image():
    # прямоугольник размером с картинку — фона не остаётся
    image = draw_rectangle(RectangleInput(width=3, height=2,
                                          image_height=2, image_width=3,
                                          shape_color=RED, background_color=BLACK))
    assert np.all(image == RED)


# ---------- draw_ellipse ----------

def make_ellipse():
    # картинка 11×11, центр (5, 5), полуоси 3 по x и 2 по y
    return draw_ellipse(EllipseInput(semi_axis_x=3, semi_axis_y=2,
                                     image_height=11, image_width=11,
                                     shape_color=RED, background_color=BLACK))


def test_ellipse_shape():
    assert make_ellipse().shape == (11, 11, 3)


def test_ellipse_center():
    assert np.array_equal(make_ellipse()[5, 5], RED)


def test_ellipse_edges_x():
    image = make_ellipse()
    assert np.array_equal(image[5, 8], RED)    # x = 5 + 3 — ровно на границе (= 1)
    assert np.array_equal(image[5, 9], BLACK)  # на шаг дальше — снаружи


def test_ellipse_edges_y():
    image = make_ellipse()
    assert np.array_equal(image[7, 5], RED)    # y = 5 + 2 — ровно на границе
    assert np.array_equal(image[8, 5], BLACK)  # дальше — снаружи


def test_ellipse_corner_is_background():
    assert np.array_equal(make_ellipse()[0, 0], BLACK)

def analyze(values, window=1):
    return analyze_time_series(TimeSeriesInput(values=values, window=window))


# ---------- среднее, дисперсия, отклонение ----------

def test_ts_mean_variance_std():
    # классический пример: среднее 5, дисперсия 4, отклонение 2
    result = analyze([2, 4, 4, 4, 5, 5, 7, 9])
    assert result.mean == pytest.approx(5)
    assert result.variance == pytest.approx(4)
    assert result.std == pytest.approx(2)


def test_ts_constant_series():
    # все значения одинаковые — разброса нет
    result = analyze([3, 3, 3, 3])
    assert result.mean == pytest.approx(3)
    assert result.variance == pytest.approx(0)
    assert result.std == pytest.approx(0)


# ---------- локальные экстремумы ----------

def test_ts_extrema_example():
    result = analyze([1, 3, 2, 5, 4, 4, 6])
    assert result.local_maxima_indices == [1, 3]
    assert result.local_minima_indices == [2]


def test_ts_plateau_is_not_extremum():
    # 2, 2 — соседи равны, «строго больше» не выполняется
    result = analyze([1, 2, 2, 1])
    assert result.local_maxima_indices == []
    assert result.local_minima_indices == []


def test_ts_ends_are_not_extrema():
    # 5 по краям больше соседа, но края не считаются
    result = analyze([5, 1, 5])
    assert result.local_maxima_indices == []
    assert result.local_minima_indices == [1]


def test_ts_monotonic_no_extrema():
    result = analyze([1, 2, 3, 4, 5])
    assert result.local_maxima_indices == []
    assert result.local_minima_indices == []


# ---------- скользящее среднее ----------

def test_ts_moving_average_example():
    result = analyze([1, 3, 2, 5, 4, 4, 6], window=3)
    assert np.allclose(result.moving_average, [2, 10 / 3, 11 / 3, 13 / 3, 14 / 3])


def test_ts_moving_average_length():
    # длина результата n - p + 1 = 10 - 4 + 1 = 7
    result = analyze(list(range(10)), window=4)
    assert len(result.moving_average) == 7


def test_ts_moving_average_window_one():
    # окно 1 — каждое среднее равно самому элементу
    result = analyze([4, 8, 15], window=1)
    assert np.allclose(result.moving_average, [4, 8, 15])


def test_ts_moving_average_full_window():
    # окно во весь ряд — одно число, среднее всего ряда
    result = analyze([2, 4, 6], window=3)
    assert np.allclose(result.moving_average, [4])

def test_one_hot_example():
    result = one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=None))
    assert np.array_equal(result, np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1],
                                            [1, 0, 0, 0]]))


def test_one_hot_class_count_bigger():
    # class_count задан больше, чем нужно — лишние столбцы из нулей
    result = one_hot(OneHotInput(labels=[0, 1], class_count=4))
    assert np.array_equal(result, np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]]))
    assert result.shape == (2, 4)


def test_one_hot_class_count_exact():
    # class_count совпадает с max + 1 — результат как без него
    with_count = one_hot(OneHotInput(labels=[0, 2, 1], class_count=3))
    without_count = one_hot(OneHotInput(labels=[0, 2, 1], class_count=None))
    assert np.array_equal(with_count, without_count)


def test_one_hot_single_label():
    result = one_hot(OneHotInput(labels=[2], class_count=None))
    assert np.array_equal(result, np.array([[0, 0, 1]]))


def test_one_hot_same_labels():
    # метка 0 ни разу не встретилась — её столбец всё равно есть, из нулей
    result = one_hot(OneHotInput(labels=[1, 1, 1], class_count=None))
    assert np.array_equal(result, np.array([[0, 1],
                                            [0, 1],
                                            [0, 1]]))


def test_one_hot_one_per_row():
    # в каждой строке ровно одна единица
    result = one_hot(OneHotInput(labels=[3, 0, 4, 1, 1, 2], class_count=None))
    assert np.all(result.sum(axis=1) == 1)