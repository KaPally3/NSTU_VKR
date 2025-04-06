# Моделирование мультипризматических рентгеновских линз:
# влияние дефектов материала на оптические свойства линз
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tkinter
from scipy.optimize import curve_fit
import warnings


res = 1000
xmin, xmax = -0.3, 0.3

PI = np.pi
EN = 30
LAMBDA = 1e-6 * 1239.8 / (EN * 1e3)
K = 2 * PI / LAMBDA
MU = 3.0235443337462318
BETA = MU * LAMBDA / (4 * PI)
DELTA = 5.43e-4 / (EN**2)
FOCAL = 8000
THETA = 130


def get_screen_size():

    root = tkinter.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    return width, height


def read_coordinates(file_path):

    coordinates = pd.read_csv(file_path, header=None)

    x = coordinates[0].values
    y = coordinates[1].values * 1e-3

    return x, y


def rotate_graphic(angle_deg, x, y):

    angle = -np.radians(angle_deg)

    pivot = np.array([x[0], y[0]], dtype=np.float64)

    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )

    translated_coordinates = np.vstack((x, y)) - pivot[:, np.newaxis]

    rotated_coordinates = rotation_matrix @ translated_coordinates

    rotated_coordinates += pivot[:, np.newaxis]

    x_rotated = rotated_coordinates[0, :]
    y_rotated = rotated_coordinates[1, :]

    return x_rotated, y_rotated


# Рассчет пройденного пути в материале
def covered_in_the_material(x_start, y_start, length, x_teeth, y_teeth, num_points):

    x_coords = np.linspace(x_start, length, num_points)
    y_coords = 0 * x_coords + y_start

    y_teeth_interp = np.interp(x_coords, x_teeth, y_teeth)

    plt.plot(x_coords, y_teeth_interp)
    plt.xlim(0, 10)
    # plt.show()
    # sys.exit(1)

    path = 0.0
    is_inside = False
    start_inside = -1
    end_inside = -1

    for i in range(len(x_coords)):
        if y_coords[i] >= y_teeth_interp[i]:

            if not is_inside:
                is_inside = True
                start_inside = i

        else:

            if is_inside:
                is_inside = False
                end_inside = i

                path += np.sqrt(
                    (x_coords[end_inside] - x_coords[start_inside]) ** 2
                    + (y_coords[end_inside] - y_coords[start_inside]) ** 2
                )

    if is_inside:
        end_inside = len(x_coords) - 1
        path += np.sqrt(
            (x_coords[end_inside] - x_coords[start_inside]) ** 2
            + (y_coords[end_inside] - y_coords[start_inside]) ** 2
        )

    return path


# функция окружности
def circle_arc(x, radius, x_center, y_center):

    arg = radius ** 2 - (x - x_center) ** 2
    # Вместо 0 должно быть nan
    #y = np.where(arg >= 0, y_center + np.sqrt(arg), 0)
    # nan_mask = np.isnan(y)

    #return  y  # y[~nan_mask]

    if not np.all(arg >= 0):
        for i in range(len(arg)):
            if arg[i] < 0:
                arg[i] = 0

    return y_center + np.sqrt(arg)


# оценка внутреннего и внешнего радиусов
def estimate_radii(x_data, y_data, start_index, end_index):

    x_segment = x_data[start_index:end_index]
    y_segment = y_data[start_index:end_index]

    x_center_guess = np.mean(x_segment)
    y_center_guess = np.max(y_segment)
    radius_guess = np.ptp(x_segment)

    try:
        popt, pcov = curve_fit(circle_arc, x_segment, y_segment, p0=[radius_guess, x_center_guess, y_center_guess], bounds=([0, min(x_data), min(y_data)], [np.max(x_data) - np.min(x_data), max(x_data), max(y_data) * 2]), method='trf')
        radius, x_center, y_center = popt

        return radius, x_center, y_center
    except RuntimeError:
        print("Error: curve_fit не смог сойтись\nПопробуйте другие начальные значения")
        return None, None, None


def find_first_close_enough(data, value):

    for d in data:
        if d >= value:
            return list(data).index(d)

    print(f"{value} не на координатной прямой")
    return None


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=UserWarning)

    a = np.finfo(np.longdouble)
    print(a.min)

    source_lens_distance = 2 * FOCAL
    lens_screen_distance = 2 * FOCAL

    xs = np.linspace(xmin, xmax, res)
    dx = (xmax - xmin) / (res - 1)

    fx = np.fft.fftfreq(res, d=dx)
    fx = np.fft.fftshift(fx)

    x_screen = fx * LAMBDA * lens_screen_distance

    lens_thickness = xs**2 / (2 * FOCAL * DELTA)

    Xsc = xmax / 2
    dx = np.linspace(-Xsc, Xsc, 1000)
    list_points = []
    for i in dx:
        formula_one = (
            EN
            * np.exp(1j * K * np.sqrt((xs - i) ** 2 + source_lens_distance**2))
            / np.sqrt((xs - i) ** 2 + source_lens_distance**2)
            * np.exp(-1j * K * (DELTA - 1j * BETA) * lens_thickness)
        )

        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(xs, np.abs(formula_one))
        # ax2.plot(xs, np.angle(formula_one))
        # plt.show()
        # exit(1)
        # print(np.abs(np.diff(formula_one)))

        formula_two = np.exp(PI * 1j / (LAMBDA * lens_screen_distance) * (xs**2))

        formula_three = (
            -1j
            / LAMBDA
            * np.exp(
                2
                * PI
                * 1j
                / LAMBDA
                * (lens_screen_distance + (x_screen**2) / 2 / lens_screen_distance)
            )
        )

        out_source = np.fft.fft(formula_one * formula_two)
        out_source = formula_three * np.fft.fftshift(out_source)

        out_source_abs = np.abs(out_source)
        list_points.append(np.argmax(out_source_abs))

        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(x_screen, np.abs(out_source))
        # ax2.plot(x_screen, np.angle(out_source))

    # fig, ax3 = plt.subplots()
    # ax3.plot(dx, list_points)
    # plt.show()

    file_path = "line.csv"
    x, y = read_coordinates(file_path)
    x_rotated, y_rotated = rotate_graphic(1, x, y)

    width, heigth = get_screen_size()
    dpi = 110
    fig, ax1 = plt.subplots(figsize=(int(width / dpi), int(heigth / dpi)))
    (line1,) = ax1.plot(x, y, color="blue", linestyle="-", label="teeth")
    ax1.plot(x_rotated, y_rotated, color="red")
    ax1.axis("equal")

    line2 = Line2D(
        [0], [0], color="green", linestyle="--", linewidth=2, label="half-period"
    )

    x_lim = np.max(x)
    period = 1
    half_period = period / 2

    for i in range(int(x_lim * period / half_period) + 1):
        x_pos = (i + 0.1) * half_period
        ax1.axvline(x=x_pos, color="green", linestyle="--", linewidth=2)

    plt.grid(True)
    plt.xlim(0, x_lim / 10)
    ax1.legend(handles=[line1, line2])
    # plt.show()

    # Генерация горизонтальных лучей
    y_rays = np.linspace(-2.5, 1.5, 500)
    list_paths = []
    for j in range(len(xs)):
        # list_paths.append(covered_in_the_material(0, y_rays[j], 100, x, y, 1000))
        list_paths.append(
            covered_in_the_material(0, xs[j], 100, x_rotated, y_rotated, 1000)
        )

    print(*list(map(float, list_paths)), sep="\n")
    # считаем фокус модельной "идеальной" параболы y_t * y_g / L
    fdist = 0.7 * 1.75 / 100
    offset = 0.3

    plt.figure()
    # собираем всю числовую прямую
    xs_combined = np.concatenate((-xs + 0.6, xs))
    list_paths_combined = np.concatenate((np.max(list_paths) - list_paths, np.max(list_paths) - list_paths))
    # сортируем по возрастанию
    sorted_indices = np.argsort(xs_combined)
    xs_combined_sorted = xs_combined[sorted_indices]
    list_paths_combined_sorted = list_paths_combined[sorted_indices]
    # отрисовываем графики нашей и "идеальной" параболы
    plt.plot(xs_combined_sorted, list_paths_combined_sorted)
    plt.plot(xs, (xs - offset) ** 2 / (2 * fdist))
    plt.axvspan(
        offset,
        offset - 0.7,
        alpha=0.2,
    )
    plt.show()

    # Оцениваем радиусы
    # Внутренний
    start_index_inner = find_first_close_enough(x, 2 * half_period - half_period / 2)
    end_index_inner = find_first_close_enough(x, 2 * half_period + half_period / 2)
    print(x[start_index_inner], x[end_index_inner])
    radius_inner, x_center_inner, y_center_inner = estimate_radii(x, y, start_index_inner, end_index_inner)
    if radius_inner is not None:
        print(f"Inner Circle\nRadius: {radius_inner} Center of circle: x = {x_center_inner} y = {y_center_inner}")

    # Внешний
    start_index_outer = find_first_close_enough(x, 3 * half_period - half_period / 2)
    end_index_outer = find_first_close_enough(x, 3 * half_period + half_period / 2)
    print(x[start_index_outer], x[end_index_outer])
    radius_outer, x_center_outer, y_center_outer = estimate_radii(x, y, start_index_outer, end_index_outer)
    if radius_outer is not None:
        print(f"Outer Circle\nRadius: {radius_outer} Center of circle: x = {x_center_outer} y = {y_center_outer}")

