import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

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

source_lens_distance = 2 * FOCAL
lens_screen_distance = 2 * FOCAL

xs = np.linspace(xmin, xmax, res)
dx = (xmax - xmin) / (res - 1)

fx = np.fft.fftfreq(res, d=dx)
fx = np.fft.fftshift(fx)

x_screen = fx * LAMBDA * lens_screen_distance

lens_thickness = xs**2 / (2 * FOCAL * DELTA)

formula_one = (
    EN
    * np.exp(1j * K * np.sqrt(xs**2 + source_lens_distance**2))
    / np.sqrt(xs**2 + source_lens_distance**2)
    * np.exp(-1j * K * (DELTA - 1j * BETA) * lens_thickness)
)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(xs, np.abs(formula_one))
ax2.plot(xs, np.angle(formula_one))
plt.show()

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


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x_screen, np.abs(out_source))
ax2.plot(x_screen, np.angle(out_source))
plt.show()
