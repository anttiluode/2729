"""Gate 0: both parents survive the merge.

(a) With the pairs disconnected and the springs fixed by hand, TwoBand's cavity is
    WallMemoryLaw's cavity.run (itself Kavitaatio's simulate() to 3e-15).
(b) A pair in TwoBand is theta_neuron.py's pair: with no cavity input it stays locked
    and D does not move; driven directly with theta_neuron's pulse it fires at amp 0.06
    and not at 0.04 (T2), and the slip moves D.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twoband import TwoBand, make_pair, wrap, KAPPA, CARRIER, DRIVE_XY  # noqa: E402
from wml.cavity import Cavity  # noqa: E402
from chains.fork import fork_world  # noqa: E402
from chains.probing import source_kernel  # noqa: E402


def main(out="results/gate0.json"):
    res = {}
    proto = make_pair()
    # (a) cavity identity, three wall words, all three sources
    cav = Cavity(K=8)
    errs = []
    for word in ([0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]):
        bits8 = np.zeros(8)
        for b, k in zip(word, (0, 2, 4, 6)):
            bits8[k] = b
        for s in range(3):
            ref = cav.run(bits8[None], source=s, steps=350)[0, :, 0]
            tw = TwoBand(couple_hear=False, _proto=proto)
            tw.kappa_override = np.array(word, float) * KAPPA
            tr = tw.ping(s, 350)
            errs.append(float(np.abs(tr - ref).max() / np.abs(ref).max()))
    res["cavity_max_rel_err"] = max(errs)
    print(f"(a) cavity vs WallMemoryLaw run: max relative error {max(errs):.2e}")

    # (b1) a pair left alone stays locked and keeps D
    tw = TwoBand(couple_hear=False, _proto=proto)
    D_start = tw.D().copy()
    tw.run(3000)
    res["idle_3000_dD_max"] = float(np.abs(tw.D() - D_start).max())
    res["idle_3000_slips"] = tw.slips().tolist()
    print(f"(b1) idle 3000 steps: |dD| max {res['idle_3000_dD_max']:.2e}, slips {res['idle_3000_slips']}")

    # (b2) theta_neuron's own drive into one pair's private field
    rows = []
    for amp in (0.04, 0.06, 0.2):
        tw = TwoBand(couple_hear=False, _proto=proto)
        k = source_kernel(tw.pairs[0], DRIVE_XY, 1.0)
        for i in range(2500):
            a = amp * math.cos(0.11 * tw.pairs[0].t) if 100 <= i < 180 else 0.0
            tw.pairs[0].step(external_source=k * a if a else None)
            for p in tw.pairs[1:]:
                p.step()
            tw.shadow.step()
            th = np.stack([p.theta.numpy() for p in tw.pairs])
            tw.unwrapped += (th - tw._prev + math.pi) % (2 * math.pi) - math.pi
            tw._prev = th.copy()
            tw.t += 1
        rows.append({"amp": amp, "slips": int(tw.slips()[0]), "dD": float(tw.D()[0])})
        print(f"(b2) direct drive amp {amp}: slips {rows[-1]['slips']}, dD {rows[-1]['dD']:+.3f} rad")
    res["direct_drive"] = rows
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
