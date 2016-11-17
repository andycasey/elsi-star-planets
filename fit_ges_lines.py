
"""
Fit a slope to the abundances relative to condensation temperatures for all
cluster stars in the Gaia-ESO Survey iDR4.
"""

import cPickle as pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.ticker import MaxNLocator

from smh.photospheres.abundances import asplund_2009 as solar_abundance


MIN_ELEMENTS = 5

# Use Lodders et al. (2003) condensation sequence temperatures
with open("lodders-2003.pkl", "rb") as fp:
    tc = pickle.load(fp)

stars = Table.read("ges-idr4-cluster-catalog.fits")

# Just keep cluster stars
stars = stars[(stars["MEMBERSHIP"] != "FIELD") * (stars["TEFF"] < 7000)]

# Only main-sequence stars
#stars = stars[stars["LOGG"] > 3.5]

species = [name for name in stars.dtype.names \
    if "NN_{}".format(name) in stars.dtype.names \
    and name not in ("TEFF", "LOGG", "XI", "MH", "FEH", "FE1", "FE2", "ALPHA_FE", "C_C2", "N_CN", "SI3") \
    and np.any(np.isfinite(stars[name]))]
elements = [each.rstrip("123") for each in species]
unique_elements = list(set(elements))

# If there are duplicate ionization stages for species, take the most populous one.
remove = []
for element in unique_elements:
    if elements.count(element) > 1:

        # Count number of finite things
        neutral = np.sum(np.isfinite(stars["{}1".format(element)]))
        ionised = np.sum(np.isfinite(stars["{}2".format(element)]))

        if neutral > ionised:
            remove.append("{}2".format(element))

        else:
            remove.append("{}1".format(element))


species = list(set(species).difference(remove))

# Subtract Solar abundances and create new columns.
elements = []
for specie in species:
    element = specie.rstrip("12").title()
    stars[element] = stars[specie] - solar_abundance(element)
    stars["e_{}".format(element)] = stars["E_{}".format(specie)]
    elements.append(element)

# We ignore these elements and will fit above ~1300 K: "Mg", "Si", "Fe"
elements = list(set(elements).difference(["Mg", "Si", "Fe", "He"]))
elements = np.array([element for element in elements if tc[element] > 1300])

# Fit [X/Fe] abundances as a function of condensation temperature.

slopes = []
N = len(stars)
keep = np.ones(N, dtype=bool)

for i, star in enumerate(stars):

    # Just close all but the last one.
    plt.close("all")

    # Get the [X/H] abundance ratios (y-axis values)
    x = np.array([tc[element] for element in elements])

    y = np.array([star[element] for element in elements])
    yerr = np.array([star["e_{}".format(element)] for element in elements])

    finite = np.isfinite(y * yerr)
    x = x[finite]
    y = y[finite]
    yerr = yerr[finite]

    print(i, N, star["CNAME"], y.size)

    if y.size < MIN_ELEMENTS:
        keep[i] = False
        continue

    # Fit a line to these refractory lithophiles
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    theta = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    #b, m

    assert np.all(np.isfinite(theta))


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
    for j, element in enumerate(elements[finite]):
        ax.text(x[j], y_text, r"${\rm " + element + "}$", color="#000000")


    ax.set_title(
        r"${twomass_id}$ $({cluster})$ $T_{{\rm eff}}$ $=$ ${teff:.0f}$ ${{\rm K}},$ $\log{{g}}$ $=$ ${logg:.3f}$".format(
        twomass_id=star["CNAME"], cluster=star["MEMBERSHIP"], teff=star["TEFF"],
        logg=star["LOGG"]),
        fontsize=12)

    # Update styles.
    ax.set_xlim(xi)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    
    ax.set_xlabel(r"${\rm Condensation}$ ${\rm temperature,}$ $T_c$ $[{\rm K}]$")
    ax.set_ylabel(r"$[{\rm X/H}]$")

    slopes.append(theta[1])

# Ignore stars where we didn't have enough abundances
stars = stars[keep]




slopes = np.array(slopes)

# Plot H-R diagram
fig, ax = plt.subplots()
scat = ax.scatter(stars["TEFF"], stars["LOGG"], c=stars["FEH"],
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



# Plot distribution of slopes
fig, ax = plt.subplots()
ax.hist(1000 * slopes, bins=100, facecolor="#666666", alpha=0.5)
ax.axvline(0, c="#666666", linestyle="--", lw=2, zorder=-1)

x_range = np.max(np.abs(ax.get_xlim()))
ax.set_xlim(-x_range, +x_range)
ax.set_xlabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

fig.tight_layout()


# Get cluster colors, etc.
cluster_names = sorted(list(set(stars["MEMBERSHIP"])))
stars["cluster_index"] = [cluster_names.index(name) for name in stars["MEMBERSHIP"]]

#cmap = mpl.colors.LinearSegmentedColormap("Accent", len(cluster_names))
from matplotlib import cm
cmap = cm.get_cmap("Accent", len(cluster_names))


# Plot slopes wrt stellar metallicity
fig, ax = plt.subplots()
scat = ax.scatter(stars["FEH"], 1000 * slopes, s=50, c=stars["cluster_index"],
    cmap=cmap)

ax.set_xlabel(r"${\rm Stellar}$ ${\rm metallicity},$ $[{\rm Fe/H}]$")
ax.set_ylabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.axhline(0, c="#666666", linestyle="--", lw=2, zorder=-1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ignore = []
for i, cluster_name in enumerate(cluster_names):
    ignore.append(ax.scatter(np.median(stars["FEH"]), [0], s=50, facecolor=cmap(i), label=cluster_name))

legend = plt.legend(frameon=False, loc="lower left")
_ = [each.set_visible(False) for each in ignore]
fig.tight_layout()




# Plot slopes wrt effective temperature
fig, ax = plt.subplots()
scat = ax.scatter(stars["TEFF"], 1000 * slopes, s=50, c=stars["cluster_index"],
    cmap=cmap)

ax.set_xlabel(r"${\rm Stellar}$ ${\rm effective}$ ${\rm temperature},$ $T_{\rm eff}$ $[{\rm K}]$")
ax.set_ylabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.axhline(0, c="#666666", linestyle="--", lw=2, zorder=-1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ignore = []
for i, cluster_name in enumerate(cluster_names):
    ignore.append(ax.scatter(np.median(stars["TEFF"]), [0], s=50, facecolor=cmap(i), label=cluster_name))

legend = plt.legend(frameon=False, loc="lower left")
_ = [each.set_visible(False) for each in ignore]
fig.tight_layout()



# Plot slopes wrt surface gravity
fig, ax = plt.subplots()
scat = ax.scatter(stars["LOGG"], 1000 * slopes, s=50, c=stars["cluster_index"],
    cmap=cmap)

ax.set_xlabel(r"${\rm Stellar}$ ${\rm surface}$ ${\rm gravity},$ $\log{g}$ $[{\rm K}]$")
ax.set_ylabel(r"${\rm Slope}$ ${\rm of}$ ${\rm refractory}$ ${\rm lithophile}$ ${\rm abundances},$ $S$ $[10^{-3}$ ${\rm dex/K}]$")
ax.axhline(0, c="#666666", linestyle="--", lw=2, zorder=-1)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ignore = []
for i, cluster_name in enumerate(cluster_names):
    ignore.append(ax.scatter(np.median(stars["LOGG"]), [0], s=50, facecolor=cmap(i), label=cluster_name))

legend = plt.legend(frameon=False, loc="lower left")
_ = [each.set_visible(False) for each in ignore]
fig.tight_layout()



raise a
