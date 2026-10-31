"""Fast checks (about a minute): the merge changes neither parent."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twoband import TwoBand, make_pair, KAPPA  # noqa: E402
from wml.cavity import Cavity  # noqa: E402

PROTO = make_pair()


def test_cavity_is_wallmemorylaw():
    bits8 = np.zeros(8)
    bits8[[0, 4]] = 1
    ref = Cavity(K=8).run(bits8[None], source=1, steps=200)[0, :, 0]
    tw = TwoBand(couple_hear=False, _proto=PROTO)
    tw.kappa_override = np.array([1, 0, 1, 0]) * KAPPA
    assert np.array_equal(tw.ping(1, 200), ref)


def test_idle_pair_keeps_D_and_never_slips():
    tw = TwoBand(couple_hear=False, _proto=PROTO)
    tw.run(1000)
    assert np.all(tw.D() == 0) and np.all(tw.slips() == 0)


def test_unwritten_wall_is_neumann():
    tw = TwoBand(_proto=PROTO)
    assert np.all(tw.kappa() == 0)


def test_fork_is_independent():
    tw = TwoBand(g_in=0.006, _proto=PROTO)
    tw.run(50)
    f = tw.fork()
    f.u = f.u + f.pulse(0)
    f.run(100)
    tw.run(100)
    assert not np.allclose(f.u, tw.u)
    assert np.all(tw.D() == 0)
