import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

res = 4000

xmin, xmax = -10, 10
ymin, ymax = -10, 10

sxmin, sxmax = -0.15, 0.15
symin, symax = -0.3, 0.3

wl = 1.0e-2

dz = 100.0

PI = np.pi
ENERGY = 30
LIMBDA = 1e-6 * 1239.8 / (ENERGY * 1e3)
K = 2 * PI / LIMBDA
MU = 3.0235443337462318
BETA = MU * LIMBDA / (4 * PI)
DELTA = 5.43e-4 / (ENERGY**2)
FOCAL = 8000
THETA = 130


source = np.zeros((res, res), dtype=np.complex128)

xs, ys = np.meshgrid(np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
dx = (xmax - xmin) / (res - 1)
dy = (ymax - ymin) / (res - 1)

source[(xs > sxmin) & (xs < sxmax) & (ys > symin) & (ys < symax)] = 1

sxmax = np.sum((xs > sxmin) & (xs < sxmax)) / res * dx / 2
sxmin = -sxmax
symax = np.sum((ys > symin) & (ys < symax)) / res * dy / 2
symin = -symax
print(sxmin, sxmax, symin, symax)

fx = np.fft.fftfreq(res, d=dx)
fy = np.fft.fftfreq(res, d=dy)
fx = np.fft.fftshift(fx)
fy = np.fft.fftshift(fy)

x_screen = fx * wl * dz
y_screen = fy * wl * dz

formula_one = 1  # (ENERGY * np.exp(1j * K * np.sqrt(xs2 + ys2)) / np.sqrt(xs2 + ys2)) * np.exp(-1j * K * (DELTA - 1j * BETA) * ys**2 / (2 * FOCAL * DELTA))

formula_two = np.exp(PI * 1j / (wl * dz) * (xs**2 + ys**2))

formula_three = (
    -1j / wl * np.exp(2 * PI * 1j / wl * (dz + (x_screen**2 + y_screen**2) / 2 / dz))
)

out_source = np.fft.fft2(source * formula_one * formula_two)
out_source = formula_three * np.fft.fftshift(out_source)


a = 1  # * (sxmax - sxmin)
formula_exp_x = a * np.sinc((sxmax - sxmin) * x_screen / (wl * dz))
formula_exp_y = a * np.sinc((symax - symin) * y_screen / (wl * dz))

norm_x = np.abs(out_source).sum(axis=0)
norm_x /= np.max(norm_x)

plt.figure(figsize=(10, 10))
plt.plot(x_screen, norm_x, "b.", x_screen, np.abs(formula_exp_x), "r-")

plt.show()
