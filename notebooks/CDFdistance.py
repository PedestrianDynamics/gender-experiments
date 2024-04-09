import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def CalcBiVarCDF(x, y, xGrid, yGrid):
    """
    Calculate the bivariate CDF of two given input signals on predefined grids.
    input:
      - x: array of size n1
      - y: array of size n2
      - xGrid: array of size m1
      - yGrid: array of size m2
    output:
      - CDF2D: matrix
    """
    nPoints = np.size(x)
    xGridLen = np.size(xGrid)
    yGridLen = np.size(yGrid)
    CDF2D = np.zeros([xGridLen, yGridLen])
    for i in range(xGridLen):
        for j in range(yGridLen):
            for k in range(nPoints):
                if (x[k] <= xGrid[i]) and (y[k] <= yGrid[j]):
                    CDF2D[i, j] += 1

    CDF2D = CDF2D / nPoints
    return CDF2D


def CDFDistance0(x1, y1, x2, y2):
    """
    For two input 2D signals calculate the distance between their CDFs.
    input:
      - x1: array of size n
      - y2: array of size n
      - x2: array of size m
      - y2: array of size m
    output:
      - KSD: not negative number
    """
    xPoints = 100
    yPoints = 100
    x = np.hstack((x1, x2))
    xCommonGrid = np.linspace(np.min(x), np.max(x), xPoints)
    y = np.hstack((y1, y2))
    yCommonGrid = np.linspace(np.min(y), np.max(y), yPoints)
    CDF1 = CalcBiVarCDF(x1, y1, xCommonGrid, yCommonGrid)
    CDF2 = CalcBiVarCDF(x2, y2, xCommonGrid, yCommonGrid)
    #    KSD = np.linalg.norm(CDF1-CDF2); # Frobenius norm (p=2)
    KSD = np.max(np.abs(CDF1 - CDF2))
    # Kolmogorov-Smirnov distance (p=inf)
    return KSD


def KSD_distance(rho1, v1, rho2, v2, rho_min, rho_max, nBins=10):
    """
    For two input 2D signals calculate the "distance" between their CDFs -
    averaged (over density bins) distance between two 1D CDFs of speed
    calculated for specific density bin.
    input:
      - rho1: density array of size n
      - v1: speed array of size n
      - rho2: density array of size m
      - v2: speed array of size m
      - rho_min: lower boundary of density value considered (used for bins creation)
      - rho_max: upper boundary of density value considered (used for bins creation)
    output:
      - KSD: not negative number from 0 to 1
    """
    EMPTY = -1
    bins = np.linspace(rho_min, rho_max, nBins + 1)
    #    dist1D = EMPTY*np.ones((1,nBins));
    dist1D = []

    for iBin in range(nBins):
        v1_b = v1[(rho1 >= bins[iBin]) * (rho1 <= bins[iBin + 1])]
        v2_b = v2[(rho2 >= bins[iBin]) * (rho2 <= bins[iBin + 1])]

        if (len(v1_b) > 0) and (len(v2_b) > 0):
            [ks2stat, p] = stats.ks_2samp(v1_b, v2_b)
            dist1D.append(ks2stat)

    # KSD = np.sum(dist1D[dist1D != EMPTY])/len(dist1D[dist1D != EMPTY]);
    KSD = np.sum(dist1D) / len(dist1D)
    return KSD


import numpy as np
from scipy import stats


def KSD_ci(rho1, v1, rho2, v2, rho_min, rho_max, n_bootstrap=100):

    nBins = 10

    bins = np.linspace(rho_min, rho_max, nBins + 1)

    ks_distances = []

    weights = []

    bootstrapped_cis = []

    bin_centers = []

    for iBin in range(nBins):

        v1_b = v1[(rho1 >= bins[iBin]) & (rho1 < bins[iBin + 1])]

        v2_b = v2[(rho2 >= bins[iBin]) & (rho2 < bins[iBin + 1])]

        bin_weight = len(v1_b) + len(v2_b)

        if bin_weight > 0:

            ks_stat, _ = stats.ks_2samp(v1_b, v2_b)

            ks_distances.append(ks_stat)

            weights.append(bin_weight)

            bin_center = (bins[iBin] + bins[iBin + 1]) / 2

            bin_centers.append(bin_center)

            # Bootstrap for confidence interval

            bootstrap_stats = []

            for _ in range(n_bootstrap):

                resample_v1_b = (
                    np.random.choice(v1_b, size=len(v1_b), replace=True)
                    if len(v1_b) > 0
                    else v1_b
                )

                resample_v2_b = (
                    np.random.choice(v2_b, size=len(v2_b), replace=True)
                    if len(v2_b) > 0
                    else v2_b
                )

                bootstrap_stat, _ = stats.ks_2samp(resample_v1_b, resample_v2_b)

                bootstrap_stats.append(bootstrap_stat)

            lower_ci = np.percentile(bootstrap_stats, 2.5)

            upper_ci = np.percentile(bootstrap_stats, 97.5)

            bootstrapped_cis.append((lower_ci, upper_ci))

    weighted_ks_distance = np.average(ks_distances, weights=weights)

    return weighted_ks_distance, bin_centers, ks_distances, bootstrapped_cis


if __name__ == "__main__":
    # Generate some test data

    np.random.seed(42)

    N = 1000

    M = 1000

    rho1 = np.random.uniform(0, 1, N)

    v1 = np.random.normal(loc=0.5, scale=0.1, size=N)

    rho2 = np.random.uniform(0, 1, M)

    v2 = np.random.normal(loc=0.5, scale=0.1, size=M) + 0.05  # Slightly shifted

    # Call the enhanced function

    weighted_ks_distance, bin_centers, ks_distances, bootstrapped_cis = (
        KSD_ci(rho1, v1, rho2, v2, 0, 1, n_bootstrap=100)
    )

    # Plot the results

    plt.figure(figsize=(12, 6))

    # KS distances for each bin

    plt.plot(bin_centers, ks_distances, label="KS Distance", marker="o", linestyle="-")

    # Confidence intervals

    lower_cis = [ci[0] for ci in bootstrapped_cis]

    upper_cis = [ci[1] for ci in bootstrapped_cis]

    plt.fill_between(
        bin_centers,
        lower_cis,
        upper_cis,
        color="gray",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    plt.xlabel("Density Bin Center")

    plt.ylabel("KS Distance")

    plt.title("KS Distances and Confidence Intervals Across Density Bins")

    plt.legend()

    plt.grid(True)

    plt.show()
