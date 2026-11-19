"""
single_receiver_nulling_examples.py

Numerical examples of Section 5 (cases A-E, Table 1, Figs. 4 and 5) of
  V. Rabinovich, "Adaptive Interference Nulling with a Single-Receiver
  Phased Array".

For every case the program
  1) simulates the K = 1 + 3N + 9N(N-1)/2 output-power readings (4.3)
     for 2-bit phase-shifter states (0, 90, 180, 270 deg),
  2) estimates the correlation coefficients R_ik by (4.6)-(4.9),
  3) solves the linear system (4.11) and scales the weights by (4.12),
  4) evaluates the null depths (4.17), the SINR before (4.18) and after
     (4.16) adaptation, the loss relative to SINR0 (4.19) and the
     adaptation gain.
It prints one line of Table 1 per case together with the adaptive weights w_n and saves the patterns of
Fig. 4 (case B) and Fig. 5 (case E) as PNG files.

Ideal conditions: identical uncoupled isotropic elements at lambda/2
spacing, ideal 2-bit phase shifters, passive lossless channels, noise
only in the receiver, broadside desired signal, SINR0 = 10 dB,
uncorrelated interferers.

Requirements: Python 3, numpy, matplotlib.
Usage:        python single_receiver_nulling_examples.py
"""
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # save figures to files, no window
import matplotlib.pyplot as plt

SINR0_DB = 10.0                           # SINR with conventional beam, no interference
STATES_DEG = np.array([0, 90, 180, 270])  # 2-bit phase-shifter states

# ----------------------------------------------------------------------
# Array geometries (element positions in wavelengths)
# ----------------------------------------------------------------------
LIN4 = np.column_stack([np.arange(4) * 0.5, np.zeros(4)])          # linear, N = 4
gx, gy = np.meshgrid(np.arange(3) * 0.5, np.arange(3) * 0.5)
PL33 = np.column_stack([gx.ravel(), gy.ravel()])                    # planar 3 x 3

# Cases of Section 5: name, positions, interferers [theta, phi, pj/pS (dB)]
CASES = [
    ("A. Linear N = 4, one interferer",    LIN4, [(47.1, 0, 40)]),
    ("B. Linear N = 4, three interferers", LIN4, [(-40, 0, 40), (25, 0, 35), (55, 0, 30)]),
    ("C. Planar 3x3, one interferer",      PL33, [(40, 30, 40)]),
    ("D. Planar 3x3, four interferers",    PL33, [(50.7, 342.2, 40), (28.6, 341.5, 40),
                                                  (38.7, 152.4, 40), (69.7, 147.3, 40)]),
    ("E. Planar 3x3, two interferers",     PL33, [(70, 15, 40), (60, 110, 40)]),
]


def phase_factors(pos, theta_deg, phi_deg):
    """Phase factors a_n(theta, phi) of all elements, Eq. (2.2)."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    u = np.sin(th) * np.cos(ph)
    v = np.sin(th) * np.sin(ph)
    return np.exp(1j * 2 * np.pi * (pos[:, 0] * u + pos[:, 1] * v))


def phase_states(N):
    """The K distinct phase-shifter states of Section 4.2 (state numbers 0..3)."""
    states = {(0,) * N}                                   # all phases 0
    for i, k in itertools.combinations(range(N), 2):      # every pair (i, k)
        for qi in range(4):
            for qk in range(4):
                q = [0] * N
                q[i], q[k] = qi, qk
                states.add(tuple(q))                      # set removes repeats
    return sorted(states)


def run_case(pos, interferers):
    N = len(pos)
    sr2 = 1.0                                    # receiver noise sigma_r^2
    pS = 10 ** (SINR0_DB / 10) * sr2 / N**2      # SINR0 = N^2 pS / sr2
    aS = phase_factors(pos, 0, 0)                # broadside desired signal

    # True correlation coefficients R_ik, Eq. (4.4) - used only to simulate readings
    R = pS * np.outer(aS, aS.conj())
    A_int, p_int = [], []
    for th, ph, ratio_dB in interferers:
        aj = phase_factors(pos, th, ph)
        pj = pS * 10 ** (ratio_dB / 10)
        R += pj * np.outer(aj, aj.conj())
        A_int.append(aj)
        p_int.append(pj)

    # Step 1a: power readings (4.3) for the K phase states
    states = phase_states(N)
    K = len(states)                                          # Eq. (4.10)
    reading = {}
    for q in states:
        e = np.exp(1j * np.radians(STATES_DEG[list(q)]))     # e^{j phi_n}
        reading[q] = np.real(e @ R @ e.conj()) + sr2         # Eq. (4.3)

    # Step 1b: coefficients from the readings, Eqs. (4.6)-(4.9)
    Rh = np.zeros((N, N), complex)
    for i, k in itertools.combinations(range(N), 2):
        Pbar = np.zeros(4)                                   # group means (4.6)
        for D in range(4):                                   # phase difference 90°*D
            for b in range(4):                               # common phase of the pair
                q = [0] * N
                q[i], q[k] = (b + D) % 4, b
                Pbar[D] += reading[tuple(q)] / 4
        Rh[i, k] = (Pbar[0] - Pbar[2]) / 4 + 1j * (Pbar[3] - Pbar[1]) / 4   # (4.8)
        Rh[k, i] = np.conj(Rh[i, k])
    iu = np.triu_indices(N, 1)
    P0 = reading[(0,) * N]
    np.fill_diagonal(Rh, (P0 - 2 * np.sum(Rh[iu].real)) / N)                  # (4.9)

    # Step 2: weights, Eqs. (4.11)-(4.12)
    v = np.linalg.solve(Rh, aS)                              # system (4.11)
    w = v / np.max(np.abs(v))                                # scaling (4.12)

    # Step 3: null depths and SINR, Eqs. (4.13)-(4.19)
    F = lambda ww, a: ww.conj() @ a                          # pattern value (2.9)
    wc = aS                                                  # conventional beam
    GS, GSc = abs(F(w, aS)) ** 2, abs(F(wc, aS)) ** 2        # gain toward the signal
    PI = sum(p * abs(F(w, a)) ** 2 for a, p in zip(A_int, p_int))     # (4.14)
    PIc = sum(p * abs(F(wc, a)) ** 2 for a, p in zip(A_int, p_int))
    res = {
        "K": K,
        "offdiag_err": np.max(np.abs(Rh[iu] - R[iu])) / np.max(np.abs(R[iu])),
        "D_conv": [10 * np.log10(abs(F(wc, a)) ** 2 / GSc) for a in A_int],
        "D_adapt": [10 * np.log10(abs(F(w, a)) ** 2 / GS) for a in A_int],  # (4.17)
        "SINR_before": 10 * np.log10(GSc * pS / (PIc + sr2)),               # (4.18)
        "SINR_after": 10 * np.log10(GS * pS / (PI + sr2)),                  # (4.16)
        "w": w, "pos": pos, "interferers": interferers,
    }
    res["loss"] = SINR0_DB - res["SINR_after"]                              # (4.19)
    res["gain"] = res["SINR_after"] - res["SINR_before"]
    return res


def pattern_dB(res, theta_deg, phi_deg):
    """Pattern in dB relative to the desired-signal direction, conventional and adaptive."""
    pos, w = res["pos"], res["w"]
    aS = phase_factors(pos, 0, 0)
    conv, adapt = [], []
    for t in theta_deg:
        a = phase_factors(pos, t, phi_deg)
        conv.append(abs(aS.conj() @ a) ** 2 / len(pos) ** 2)
        adapt.append(abs(w.conj() @ a) ** 2 / abs(w.conj() @ aS) ** 2)
    return 10 * np.log10(conv), 10 * np.log10(adapt)


def save_figure(fig, path):
    """Save a figure; if the file is locked (e.g. open in an image viewer), save under a new name."""
    try:
        fig.savefig(path, dpi=200)
        return path
    except OSError:
        root, ext = os.path.splitext(path)
        for k in range(1, 100):
            alt = f"{root}_{k}{ext}"
            try:
                fig.savefig(alt, dpi=200)
                print(f"  note: {os.path.basename(path)} could not be overwritten "
                      f"(is it open in another program?); saved as {os.path.basename(alt)}")
                return alt
            except OSError:
                continue
        raise


def main():
    here = os.path.dirname(os.path.abspath(__file__))   # folder of this program
    f4 = os.path.join(here, "fig4_case_B.png")
    f5 = os.path.join(here, "fig5_case_E.png")
    results = []
    for name, pos, intf in CASES:
        r = run_case(pos, intf)
        results.append(r)
        print(f"\n{name}   (K = {r['K']})")
        print(f"  max relative error of off-diagonal estimates: {r['offdiag_err']:.1e}")
        print("  null depth, conventional -> adaptive:" +
              "".join(f"  {c:.1f} -> {a:.1f} dB" for c, a in zip(r["D_conv"], r["D_adapt"])))
        print(f"  SINR before -> after: {r['SINR_before']:.1f} -> {r['SINR_after']:+.1f} dB")
        print(f"  loss vs SINR0: {r['loss']:.1f} dB,  gain G: {r['gain']:.1f} dB")
        print("  adaptive weights (4.12), w_n = |w_n| exp(j arg w_n), element position x, y in wavelengths:")
        for n, (wn, p) in enumerate(zip(r["w"], r["pos"]), start=1):
            print(f"    w{n} = {abs(wn):.3f} exp(j{np.degrees(np.angle(wn)):+7.1f} deg)"
                  f"   ({wn.real:+.3f} {wn.imag:+.3f}j)   at x = {p[0]:.1f}, y = {p[1]:.1f}")

    # Fig. 4: case B, linear array
    rB = results[1]
    th = np.arange(-90, 90.01, 0.1)
    conv, adapt = pattern_dB(rB, th, 0)
    plt.figure(figsize=(7, 4.2))
    plt.plot(th, conv, "--", label="Conventional")
    plt.plot(th, adapt, "-", label="Adaptive, weights (4.11)-(4.12)")
    for t, _, _ in rB["interferers"]:
        plt.axvline(t, color="k", ls=":", lw=0.8)
    plt.xlim(-90, 90); plt.ylim(-120, 5); plt.grid(alpha=0.3)
    plt.xlabel("θ, deg"); plt.ylabel("dB relative to the desired signal")
    plt.title("Case B: linear array N = 4, three interferers")
    plt.legend(loc="lower left"); plt.tight_layout()
    f4 = save_figure(plt.gcf(), f4)

    # Fig. 5: case E, cuts through the two interferers
    rE = results[4]
    th = np.arange(0, 90.01, 0.1)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
    for ax, (tj, pj, _), name in zip(axs, rE["interferers"], ["J1", "J2"]):
        conv, adapt = pattern_dB(rE, th, pj)
        ax.plot(th, conv, "--", label="Before adaptation")
        ax.plot(th, adapt, "-", label="After adaptation")
        ax.axvline(tj, color="k", ls=":", lw=0.8)
        ax.set_xlim(0, 90); ax.set_ylim(-120, 5); ax.grid(alpha=0.3)
        ax.set_xlabel("θ, deg"); ax.set_title(f"Case E: cut ϕ = {pj}° ({name})")
    axs[0].set_ylabel("dB relative to the desired signal")
    axs[0].legend(loc="lower left"); fig.tight_layout()
    f5 = save_figure(fig, f5)
    print(f"\nSaved:\n  {f4}\n  {f5}")


if __name__ == "__main__":
    main()
