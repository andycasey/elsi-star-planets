
"""
Fit a slope to the abundances relative to condensation temperatures for all
stars in the Ness/APOGEE cluster sample.
"""

import cPickle as pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.ticker import MaxNLocator


# Use Lodders et al. (2003) condensation sequence temperatures
with open("lodders-2003.pkl", "rb") as fp:
    tc = pickle.load(fp)

# Load in Melissa's values.
stars = Table.read("Ness-APOGEE-clusters.fits")
elements = [
    "Na", "Rb", "Ti", "C", "K", "Mn", "O", "N", "P", "S", "V", "Co", "Mg", "Si",
    "Ca", "Al", "Cr", "Cu"] 

# Convert MKN's [X/Fe] abundance ratios to [X/H]
for element in elements:
    stars[element] = stars[element] + stars["[Fe/H]"]

# We ignore these elements and will fit above ~1300 K: "Mg", "Si", "Fe"
elements = list(set(elements).difference(["Mg", "Si", "Fe"]))
elements = [element for element in elements if tc[element] > 1300]

# As per MKN advice, the elemental errors are taken from cross-validation and
# are therefore homoscedastic over 
element_errors = {
    'Al': 0.031,
    'C': 0.04,
    'Ca': 0.021,
    'Co': 0.072,
    'Cr': 0.027,
    'Cu': 0.076,
    'Fe': 0.021,
    'K': 0.025,
    'Mg': 0.022,
    'Mn': 0.02,
    'N': 0.052,
    'Na': 0.093,
    'Ni': 0.015,
    'O': 0.038,
    'P': 0.119,
    'Rb': 0.107,
    'S': 0.075,
    'Si': 0.022,
    'Ti': 0.037,
    'V': 0.074
}


# Fit [X/Fe] abundances as a function of condensation temperature.
x = np.array([tc[element] for element in elements])

# Reverse it
stars = stars[::-1]

slopes = np.zeros(len(stars))

for i, star in enumerate(stars):

    # Just close all but the last one.
    plt.close("all")

    # Get the [X/H] abundance ratios (y-axis values)
    y = np.array([star[element] for element in elements])
    yerr = np.array([element_errors[element] for element in elements])

    # Fit a line to these refractory lithophiles
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    theta = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    #b, m


    fig, ax = plt.subplots()
    ax.axhline(0, c="#666666", linestyle="--", lw=2, zorder=-1)

    ax.errorbar(x, y, yerr=yerr, fmt=None, ecolor="#000000", elinewidth=2)
    ax.scatter(x, y, facecolor="#000000", s=50)

    # Draw a line and project covariance matrix.
    xi = np.array(ax.get_xlim())
    ax.plot(xi, np.polyval(theta[::-1], xi), c="r", zorder=-1, lw=2)

    yi_samples = np.array([np.polyval(theta_sample[::-1], xi) \
        for theta_sample in np.random.multivariate_normal(theta, cov, size=100)])

    yi_fill = np.array([
        np.percentile(yi_samples[:, 0], [16, 84]),
        np.percentile(yi_samples[:, 1], [16, 84])
    ])
    
    ax.fill_between(
        xi, yi_fill[:, 0], yi_fill[:, 1], facecolor="r", alpha=0.5,
        zorder=-1, edgecolor="None")

    # Text labels for the elements.
    y_text = ax.get_ylim()[0] + np.ptp(ax.get_ylim()) * 0.95
    for j, element in enumerate(elements):
        ax.text(x[j], y_text, r"${\rm " + element + "}$", color="#000000")


    ax.set_title(
        r"${twomass_id}$ $({cluster})$ $T_{{\rm eff}}$ $=$ ${teff:.0f}$ ${{\rm K}},$ $\log{{g}}$ $=$ ${logg:.3f}$".format(
        twomass_id=star["2MASSID"], cluster=star["cluster"], teff=star["teff"],
        logg=star["logg"]),
        fontsize=12)

    # Update styles.
    ax.set_xlim(xi)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    
    ax.set_xlabel(r"${\rm Condensation}$ ${\rm temperature,}$ $T_c$ $[{\rm K}]$")
    ax.set_ylabel(r"$[{\rm X/H}]$")

    print(star["2MASSID"], theta[1])

    slopes[i] = theta[1]


fig.savefig("apogee-star-tc-example.pdf", dpi=300)


# Plot H-R diagram
fig, ax = plt.subplots()
scat = ax.scatter(stars["teff"], stars["logg"], c=stars["[Fe/H]"],
    cmap="viridis", s=50)
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

cmap = plt.colorbar(scat)
cmap.set_label(r"${\rm Stellar}$ ${\rm metallicity},$ $[{\rm Fe/H}]$")

ax.set_xlabel(r"${\rm Stellar}$ ${\rm effective}$ ${\rm temperature},$ $T_{\rm eff}$ $[{\rm K}]$")
ax.set_ylabel(r"${\rm Stellar}$ ${\rm surface}$ ${\rm gravity},$ $\log{g}$")

fig.tight_layout()
fig.savefig("apogee-cluster-hrd.pdf", dpi=300)



# Plot distribution of slopes
fig, ax = plt.subplots()
ax.hist(1000 * slopes, bins=10, facecolor="#666666", alpha=0.5)
ax.axvline(0, c="#666666", linestyle="--", lw=2, zorder=-1)

x_range = np.max(np.abs(ax.get_xlim()))
ax.set_xlim(-x_range, +x_range)
ax.set_xlabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

fig.tight_layout()
fig.savefig("apogee-cluster-slope-distribution.pdf", dpi=300)



# Get cluster colors, etc.
colors = [
    # From flatuicolors.com
    "#1abc9c",
    "#2ecc71",
    "#3498db",
    "#9b59b6",
    "#34495e",
    "#f1c40f",
    "#e67e22",
    "#e74c3c",
]
cmap = mpl.colors.ListedColormap(colors)

cluster_names = sorted(list(set(stars["cluster"])))
stars["cluster_index"] = [cluster_names.index(name) for name in stars["cluster"]]



# Plot slopes wrt stellar metallicity
fig, ax = plt.subplots()
scat = ax.scatter(stars["[Fe/H]"], 1000 * slopes, s=50, c=stars["cluster_index"],
    cmap=cmap)

ax.set_xlabel(r"${\rm Stellar}$ ${\rm metallicity},$ $[{\rm Fe/H}]$")
ax.set_ylabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.axhline(0, c="#666666", linestyle="--", lw=2, zorder=-1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ignore = []
for color, cluster_name in zip(colors, cluster_names):
    ignore.append(ax.scatter(np.median(stars["[Fe/H]"]), [0], s=50, facecolor=color, label=cluster_name))

legend = plt.legend(frameon=False, loc="lower left")
_ = [each.set_visible(False) for each in ignore]
fig.tight_layout()
fig.savefig("apogee-cluster-slope-wrt-feh.pdf", dpi=300)





# Plot slopes wrt effective temperature
fig, ax = plt.subplots()
scat = ax.scatter(stars["teff"], 1000 * slopes, s=50, c=stars["cluster_index"],
    cmap=cmap)

ax.set_xlabel(r"${\rm Stellar}$ ${\rm effective}$ ${\rm temperature},$ $T_{\rm eff}$ $[{\rm K}]$")
ax.set_ylabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.axhline(0, c="#666666", linestyle="--", lw=2, zorder=-1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ignore = []
for color, cluster_name in zip(colors, cluster_names):
    ignore.append(ax.scatter(np.median(stars["teff"]), [0], s=50, facecolor=color, label=cluster_name))

legend = plt.legend(frameon=False, loc="lower left")
_ = [each.set_visible(False) for each in ignore]
fig.tight_layout()
fig.savefig("apogee-cluster-slope-wrt-teff.pdf", dpi=300)



raise a
