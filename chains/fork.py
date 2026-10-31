"""Full mutable-state fork of a World (copied from ComputationalChains experiments/bond_ablation.py)."""
import copy

import numpy as np
import torch


def fork_world(world):
    w = copy.copy(world)
    w.cfg = copy.deepcopy(world.cfg)
    for name in ("psi", "psi_prev", "pos", "theta", "typ", "nb", "held_by", "pinned", "field_force"):
        setattr(w, name, getattr(world, name).clone())
    w.gen = torch.Generator(device=world.device)
    w.gen.set_state(world.gen.get_state().clone())
    w.rng = np.random.default_rng()
    w.rng.bit_generator.state = copy.deepcopy(world.rng.bit_generator.state)
    w.held = copy.deepcopy(world.held)
    w.events = list(world.events)
    return w
