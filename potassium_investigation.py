

import smh
import matplotlib.pyplot as plt


order = smh.specutils.Spectrum1D.read("telluric-normalized-order.fits")

wavelengths = [
    7664.90,
    7698.98,
]

fig, ax = plt.subplots()
ax.plot(order.dispersion, order.flux, c="k")

for wavelength in wavelengths:
    ax.axvline(wavelength, c="b", lw=2, zorder=-1)

ax.set_xlabel(r"${\rm Wavelength}$ $[{\rm \AA}]$")
ax.set_ylabel(r"${\rm Normalized}$ ${\rm flux}$")

_ = order.dispersion[np.isfinite(order.flux)]
ax.set_xlim(_.min(), _.max())

fig.tight_layout()
fig.savefig("potassium-flux.pdf", dpi=300)



# Try radial velocities from (-500, 500) at an interval.
interval = 5
vrads = np.arange(-500, 500 + interval/2., interval)
lines = np.zeros((len(vrads), 2), dtype=bool)

# At each radial velocity, check how many of the two lines are measurable.
# Here we define measurable as *all fluxes* within 4 neighbouring pixels are 
# greater than 0.93
def is_measurable(wavelength):
    index = order.dispersion.searchsorted(wavelength)
    nearby_fluxes = order.flux[index - 4:index + 5]
    return np.all(nearby_fluxes >= 0.93)


for i, vrad in enumerate(vrads):
    for j, rest_wavelength in enumerate(wavelengths):
        observed_wavelength = rest_wavelength * (1 + vrad/299792.458)
        lines[i, j] = is_measurable(observed_wavelength)




fig, ax = plt.subplots(figsize=(8.75, 2.6))

ax.imshow(lines.T, aspect="auto", interpolation="nearest",
    extent=(vrads[0], vrads[-1], 0, len(wavelengths)),
    cmap="Accent_r")

ax.plot(vrads, np.ones(len(vrads)), lw=2, c="#000000")
ax.set_xlim(vrads[0], vrads[-1])
ax.set_ylim(0, 2)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(
    [r"${:.0f}$ ${{\rm \AA}}$".format(wavelength) for wavelength in wavelengths],
    rotation=90)


ax.set_xlabel(r"${\rm Stellar}$ ${\rm radial}$ ${\rm velocity},$ $v_r$ $[{\rm km}$ ${\rm s}^{-1}]$")
fig.tight_layout()

fig.savefig("potassium-measurable.pdf", dpi=300)




