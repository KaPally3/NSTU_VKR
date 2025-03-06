# Моделирование мультипризматических рентгеновских линз:
# влияние дефектов материала на оптические свойства линз
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tkinter

res = 4000
xmin, xmax = -0.2, 0.2

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

    # x = [1, 2, 3, 4, 5, 6]
    # y = [3, 3, 3, 3, 3, 3]

    # x = np.linspace(0, 10, 1000)
    # y = np.sin(x)

    return x, y


def rotate_graphic(angle_deg, x, y):

    angle = -np.radians(angle_deg)

    pivot = np.array([x[0], y[0]])

    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    translated_coordinates = np.vstack((x, y)) - pivot[:, np.newaxis]

    rotated_coordinates = rotation_matrix @ translated_coordinates

    rotated_coordinates += pivot[:, np.newaxis]

    x_rotated = rotated_coordinates[0, :]
    y_rotated = rotated_coordinates[1, :]

    return x_rotated, y_rotated


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
ax2 = ax1.twinx()
(line1,) = ax1.plot(x, y, color="blue", linestyle="-", label="teeth")
ax2.plot(x_rotated, y_rotated, color="red")

x_lim = 10
period = 1
half_period = period / 2

line2 = Line2D(
    [0], [0], color="green", linestyle="--", linewidth=2, label="half-period"
)

for i in range(int(x_lim * period / half_period) + 1):
    x_pos = (i + 0.1) * half_period
    ax1.axvline(x=x_pos, color="green", linestyle="--", linewidth=2)

plt.grid(True)
plt.xlim(0, x_lim)
ax1.legend(handles=[line1, line2])
plt.show()
