"""
adaptive_array_patterns.py

Pattern of a single-receiver adaptive array (method of the paper
V. Rabinovich, "Adaptive Interference Nulling with a Single-Receiver
Phased Array") for three array types:

  * linear   - N elements on the x axis
  * planar   - M x M elements in the xy plane (equal number in rows and columns)
  * circular - concentric rings in the xy plane, optional center element

and any number of interferers J.

Processing (as in Section 4 of the paper):
  1) K output-power readings (4.3) for 2-bit phase-shifter states,
  2) correlation coefficients R_ik by (4.6)-(4.9),
  3) optional DMR correction of the estimated matrix,
  4) weights by (4.11)-(4.12),
  5) null depths (4.17), SINR before (4.18) and after (4.16) adaptation.

The program prints the results and the weights, and saves figures:
  * array_geometry.png      - element positions and interferer directions
  * pattern_linear.png      - pattern versus theta            (linear array)
  * pattern_map.png         - 2-D pattern map over (u, v)     (planar / circular)
  * pattern_cuts.png        - cuts through every interferer   (planar / circular)
Figures are saved in the folder of this program.

Requirements: Python 3, numpy, matplotlib.
Usage: edit the SETTINGS block below and run
       python adaptive_array_patterns.py
"""
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# SETTINGS - change these values
# ======================================================================
GEOMETRY = "circular"         # "linear", "planar" or "circular"

N_LINEAR = 6                  # linear: number of elements
M_PLANAR = 4                  # planar: elements per row and per column (M x M)
RINGS = [(0.5, 6), (1.0, 12)] # circular: list of (ring radius in wavelengths, number of elements)
CENTER_ELEMENT = True         # circular: add an element at the center
SPACING = 0.5                 # linear / planar: element spacing in wavelengths

J = 3                         # number of interferers
INTERFERERS = None            # None = random directions; or a list of (theta, phi) in degrees,
                              #   e.g. [(40, 30), (60, 200)]; for a linear array phi = 0 or 180
ISR_DB = 40.0                 # interference-to-signal ratio of every interferer, dB
THETA_MIN = 30.0              # random directions: theta from THETA_MIN to THETA_MAX (outside main beam)
THETA_MAX = 85.0
SEED = 1                      # random-number seed (change to get other directions)

SINR0_DB = 10.0               # SINR of the conventional beam without interference
USE_DMR = True                # dominant mode rejection correction of the estimated matrix
# ======================================================================

STATES_DEG = np.array([0, 90, 180, 270])


# ----------------------------------------------------------------------
# Array geometry
# ----------------------------------------------------------------------
def element_positions():
    """Element positions (x, y) in wavelengths."""
    if GEOMETRY == "linear":
        x = (np.arange(N_LINEAR) - (N_LINEAR - 1) / 2) * SPACING
        return np.column_stack([x, np.zeros(N_LINEAR)])
    if GEOMETRY == "planar":
        g = (np.arange(M_PLANAR) - (M_PLANAR - 1) / 2) * SPACING
        gx, gy = np.meshgrid(g, g)
        return np.column_stack([gx.ravel(), gy.ravel()])
    if GEOMETRY == "circular":
        pts = [(0.0, 0.0)] if CENTER_ELEMENT else []
        for radius, n in RINGS:
            for k in range(n):
                a = 2 * np.pi * k / n
                pts.append((radius * np.cos(a), radius * np.sin(a)))
        return np.array(pts)
    raise ValueError("GEOMETRY must be 'linear', 'planar' or 'circular'")


def phase_factors(pos, theta_deg, phi_deg):
    """Phase factors a_n(theta, phi), Eq. (2.2)."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.exp(1j * 2 * np.pi * (pos[:, 0] * np.sin(th) * np.cos(ph) +
                                    pos[:, 1] * np.sin(th) * np.sin(ph)))


def interferer_directions(rng):
    """(theta, phi) of the J interferers in degrees."""
    if INTERFERERS is not None:
        return [tuple(d) for d in INTERFERERS]
    dirs = []
    for _ in range(J):
        if GEOMETRY == "linear":
            th = rng.uniform(THETA_MIN, min(THETA_MAX, 90.0))
            dirs.append((th, rng.choice([0.0, 180.0])))       # either side of broadside
        else:
            # uniform over the solid angle THETA_MIN..THETA_MAX, all azimuths
            c = rng.uniform(np.cos(np.radians(THETA_MAX)), np.cos(np.radians(THETA_MIN)))
            dirs.append((np.degrees(np.arccos(c)), rng.uniform(0, 360)))
    return dirs


# ----------------------------------------------------------------------
# Processing of Section 4
# ----------------------------------------------------------------------
def phase_states(N):
    """The K distinct phase-shifter states of Section 4.2 (state numbers 0..3)."""
    states = {(0,) * N}
    for i, k in itertools.combinations(range(N), 2):
        for qi in range(4):
            for qk in range(4):
                q = [0] * N
                q[i], q[k] = qi, qk
                states.add(tuple(q))
    return states


def estimate_R(reading, N):
    """Correlation coefficients from the power readings, Eqs. (4.6)-(4.9)."""
    Rh = np.zeros((N, N), complex)
    for i, k in itertools.combinations(range(N), 2):
        Pbar = np.zeros(4)                                      # group means (4.6)
        for D in range(4):
            for b in range(4):
                q = [0] * N
                q[i], q[k] = (b + D) % 4, b
                Pbar[D] += reading[tuple(q)] / 4
        Rh[i, k] = (Pbar[0] - Pbar[2]) / 4 + 1j * (Pbar[3] - Pbar[1]) / 4   # (4.8)
        Rh[k, i] = np.conj(Rh[i, k])
    iu = np.triu_indices(N, 1)
    np.fill_diagonal(Rh, (reading[(0,) * N] - 2 * np.sum(Rh[iu].real)) / N)  # (4.9)
    return Rh


def dmr(Rh, J):
    """Dominant mode rejection: keep the J largest eigenvalues, replace the rest by their mean."""
    lam, V = np.linalg.eigh((Rh + Rh.conj().T) / 2)
    idx = np.argsort(lam)[::-1]
    lam, V = lam[idx], V[:, idx]
    if J < len(lam):
        lam[J:] = lam[J:].mean()
    return V @ np.diag(lam) @ V.conj().T


def save_figure(fig, path):
    """Save a figure; if the file is locked (open elsewhere), save under a new name."""
    root, ext = os.path.splitext(path)
    for k in range(100):
        name = path if k == 0 else f"{root}_{k}{ext}"
        try:
            fig.savefig(name, dpi=200)
            return name
        except OSError:
            continue
    raise OSError(f"cannot save {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(SEED)

    pos = element_positions()
    N = len(pos)
    dirs = interferer_directions(rng)
    Jn = len(dirs)

    sr2 = 1.0                                       # receiver noise sigma_r^2
    pS = 10 ** (SINR0_DB / 10) * sr2 / N**2         # SINR0 = N^2 pS / sr2
    pJ = pS * 10 ** (ISR_DB / 10)
    aS = phase_factors(pos, 0, 0)                   # broadside desired signal
    aJ = [phase_factors(pos, t, p) for t, p in dirs]

    # true coefficients (4.4) - used only to simulate the readings
    R = pS * np.outer(aS, aS.conj())
    for a in aJ:
        R += pJ * np.outer(a, a.conj())

    # Step 1: K power readings (4.3) and estimates (4.6)-(4.9)
    states = phase_states(N)
    K = len(states)
    reading = {}
    for q in states:
        e = np.exp(1j * np.radians(STATES_DEG[list(q)]))
        reading[q] = np.real(e @ R @ e.conj()) + sr2
    Rh = estimate_R(reading, N)
    if USE_DMR:
        Rh = dmr(Rh, Jn)

    # Step 2: weights (4.11)-(4.12)
    v = np.linalg.solve(Rh, aS)
    w = v / np.max(np.abs(v))

    # Step 3: null depths and SINR
    F = lambda ww, a: ww.conj() @ a
    wc = aS
    GS, GSc = abs(F(w, aS)) ** 2, abs(F(wc, aS)) ** 2
    PI = sum(pJ * abs(F(w, a)) ** 2 for a in aJ)
    PIc = sum(pJ * abs(F(wc, a)) ** 2 for a in aJ)
    sinr_b = 10 * np.log10(GSc * pS / (PIc + sr2))     # (4.18)
    sinr_a = 10 * np.log10(GS * pS / (PI + sr2))        # (4.16)

    # ---------------- printout ----------------
    print(f"Geometry: {GEOMETRY},  N = {N} elements,  K = {K} power readings,  J = {Jn} interferers")
    print(f"Interference {ISR_DB:.0f} dB above the desired signal, SINR0 = {SINR0_DB:.0f} dB, "
          f"DMR correction: {'on' if USE_DMR else 'off'}")
    if Jn >= N:
        print("  warning: J >= N - an N-element array cannot place deep nulls on all interferers")
    print("\nInterferer        theta     phi    null depth: conventional -> adaptive")
    for j, ((t, p), a) in enumerate(zip(dirs, aJ), start=1):
        dc = 10 * np.log10(abs(F(wc, a)) ** 2 / GSc)
        da = 10 * np.log10(abs(F(w, a)) ** 2 / GS)
        print(f"  J{j:<3}         {t:7.1f}  {p:7.1f}      {dc:7.1f} -> {da:7.1f} dB")
    print(f"\nSINR before adaptation: {sinr_b:6.1f} dB")
    print(f"SINR after adaptation:  {sinr_a:6.1f} dB   (loss vs SINR0 {SINR0_DB - sinr_a:.1f} dB, "
          f"gain {sinr_a - sinr_b:.1f} dB)")
    print("\nAdaptive weights (4.12), w_n = |w_n| exp(j arg w_n):")
    for n, (wn, p) in enumerate(zip(w, pos), start=1):
        print(f"  w{n:<3} = {abs(wn):.3f} exp(j{np.degrees(np.angle(wn)):+7.1f} deg)"
              f"   at x = {p[0]:+.3f}, y = {p[1]:+.3f}")

    saved = []

    # ---------------- figure: geometry ----------------
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(pos[:, 0], pos[:, 1], s=40, c=np.abs(w), cmap="viridis", vmin=0, vmax=1,
               edgecolors="k", zorder=3)
    for n, p in enumerate(pos, start=1):
        ax.annotate(str(n), p, textcoords="offset points", xytext=(4, 4), fontsize=7)
    ext = np.max(np.abs(pos)) + 0.5
    for j, (t, p) in enumerate(dirs, start=1):
        u, vv = np.sin(np.radians(t)) * np.cos(np.radians(p)), np.sin(np.radians(t)) * np.sin(np.radians(p))
        ax.annotate("", xy=(u * ext, vv * ext), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="r", lw=1))
        ax.text(u * ext * 1.05, vv * ext * 1.05, f"J{j}", color="r", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-ext * 1.2, ext * 1.2); ax.set_ylim(-ext * 1.2, ext * 1.2)
    ax.set_xlabel("x, wavelengths"); ax.set_ylabel("y, wavelengths")
    ax.set_title(f"{GEOMETRY} array, N = {N}\ncolour = |w_n|, arrows = interferer azimuths", fontsize=9)
    ax.grid(alpha=0.3); fig.tight_layout()
    saved.append(save_figure(fig, os.path.join(here, "array_geometry.png"))); plt.close(fig)

    def patt(ww, th, ph):
        A = np.exp(1j * 2 * np.pi * (np.outer(np.sin(np.radians(th)) * np.cos(np.radians(ph)), pos[:, 0]) +
                                     np.outer(np.sin(np.radians(th)) * np.sin(np.radians(ph)), pos[:, 1])))
        return 10 * np.log10(np.maximum(np.abs(A @ ww.conj()) ** 2 / abs(F(ww, aS)) ** 2, 1e-15))

    if GEOMETRY == "linear":
        # ---------------- figure: pattern versus theta ----------------
        th = np.arange(-90, 90.01, 0.1)
        thp, php = np.abs(th), np.where(th < 0, 180.0, 0.0)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(th, patt(wc, thp, php), "--", label="Conventional")
        ax.plot(th, patt(w, thp, php), "-", label="Adaptive")
        for j, (t, p) in enumerate(dirs, start=1):
            ts = -t if p == 180.0 else t
            ax.axvline(ts, color="r", ls=":", lw=0.8)
            ax.text(ts, 2, f"J{j}", color="r", ha="center", fontsize=8)
        ax.set_xlim(-90, 90); ax.set_ylim(-80, 5); ax.grid(alpha=0.3)
        ax.set_xlabel("θ, deg (negative: φ = 180°)"); ax.set_ylabel("dB relative to the desired signal")
        ax.set_title(f"Linear array N = {N}, J = {Jn}, SINR {sinr_b:.1f} → {sinr_a:.1f} dB")
        ax.legend(loc="lower left"); fig.tight_layout()
        saved.append(save_figure(fig, os.path.join(here, "pattern_linear.png"))); plt.close(fig)
    else:
        # ---------------- figure: 2-D map over (u, v) ----------------
        u = np.linspace(-1, 1, 301)
        U, V = np.meshgrid(u, u)
        inside = U**2 + V**2 <= 1
        thm = np.degrees(np.arcsin(np.clip(np.sqrt(U**2 + V**2), 0, 1)))
        phm = np.degrees(np.arctan2(V, U))
        Z = np.full(U.shape, np.nan)
        Z[inside] = patt(w, thm[inside], phm[inside])
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.pcolormesh(U, V, Z, vmin=-60, vmax=0, cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, label="dB relative to the desired signal")
        for j, (t, p) in enumerate(dirs, start=1):
            uu, vv = np.sin(np.radians(t)) * np.cos(np.radians(p)), np.sin(np.radians(t)) * np.sin(np.radians(p))
            ax.plot(uu, vv, "rx", ms=8, mew=2); ax.text(uu + 0.03, vv + 0.03, f"J{j}", color="r", fontsize=9)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="k", lw=0.8))
        ax.set_aspect("equal"); ax.set_xlabel("u = sinθ cosφ"); ax.set_ylabel("v = sinθ sinφ")
        ax.set_title(f"Adaptive pattern, {GEOMETRY} array N = {N}, J = {Jn}", fontsize=10)
        fig.tight_layout()
        saved.append(save_figure(fig, os.path.join(here, "pattern_map.png"))); plt.close(fig)

        # ---------------- figure: cuts through the interferers ----------------
        th = np.arange(0, 90.01, 0.1)
        ncol = min(Jn, 3); nrow = int(np.ceil(Jn / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False, sharey=True)
        for j, (t, p) in enumerate(dirs):
            ax = axs[j // ncol][j % ncol]
            ax.plot(th, patt(wc, th, np.full_like(th, p)), "--", label="Conventional")
            ax.plot(th, patt(w, th, np.full_like(th, p)), "-", label="Adaptive")
            ax.axvline(t, color="r", ls=":", lw=0.8)
            ax.set_xlim(0, 90); ax.set_ylim(-80, 5); ax.grid(alpha=0.3)
            ax.set_title(f"cut φ = {p:.1f}° (J{j + 1} at θ = {t:.1f}°)", fontsize=9)
            ax.set_xlabel("θ, deg")
        for k in range(Jn, nrow * ncol):
            axs[k // ncol][k % ncol].axis("off")
        axs[0][0].set_ylabel("dB relative to the desired signal")
        axs[0][0].legend(loc="lower left", fontsize=8); fig.tight_layout()
        saved.append(save_figure(fig, os.path.join(here, "pattern_cuts.png"))); plt.close(fig)

    print("\nSaved:")
    for s in saved:
        print("  " + s)


if __name__ == "__main__":
    main()
