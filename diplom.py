import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

res = 40000

xmin, xmax = -0.2, 0.2
ymin, ymax = -10, 10

sxmin, sxmax = -0.15, 0.15
symin, symax = -0.3, 0.3


PI = np.pi
EN = 30
LAMBDA = 1e-6 * 1239.8 / (EN * 1e3)
K = 2 * PI / LAMBDA
MU = 3.0235443337462318
BETA = MU * LAMBDA / (4 * PI)
DELTA = 5.43e-4 / (EN**2)
FOCAL = 8000
THETA = 130

wl = LAMBDA
dz = 2 * FOCAL
source = np.ones(res, dtype=float)  # np.complex128)
xs = np.linspace(xmin, xmax, res)

dx = (xmax - xmin) / (res - 1)

fx = np.fft.fftfreq(res, d=dx)
fx = np.fft.fftshift(fx)

x_screen = fx * wl * dz

formula_one = (
    EN
    * np.exp(1j * K * np.sqrt(xs**2 + (2 * FOCAL) ** 2))
    / np.sqrt(xs**2 + (2 * FOCAL) ** 2)
    * np.exp(-1j * K * (DELTA - 1j * BETA) * xs**2 / (2 * FOCAL * DELTA))
)

plt.plot(xs, np.abs(formula_one))
# plt.plot(xs, np.angle(formula_one))
plt.show()

formula_two = np.exp(PI * 1j / (wl * dz) * (xs**2))

formula_three = (
    -1j / wl * np.exp(2 * PI * 1j / wl * (2 * FOCAL + (x_screen**2) / 2 / 2 / FOCAL))
)

out_source = np.fft.fft(source * formula_one * formula_two)
out_source = formula_three * np.fft.fftshift(out_source)
a = 1  # * (sxmax - sxmin)

plt.plot(x_screen, np.abs(out_source))
# plt.plot(x_screen, np.abs(formula_exp_x))
plt.show()

# formula_exp_y = a * np.sinc((symax - symin) * y_screen / (wl * dz))

# plt.imshow(np.abs(out_source))
# plt.show()

# # norm_x = np.abs(out_source).sum(axis=0)
# # norm_x /= np.max(norm_x)

# # plt.figure(figsize=(40, 20))
# # plt.plot(x_screen, norm_x, "b.", x_screen, np.abs(formula_exp_x), "r-")

# # plt.show()
