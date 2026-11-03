"""Shifted test sets: identity at zero shift, distinct files otherwise."""

import dataclasses

import pytest

from smi.data.frozen import frozen_set_path
from smi.experiments.grids import grid
from smi.experiments.runner import expand_grid
from smi.experiments.shift_eval import NOMINAL_SIM, WAVELENGTH_PATTERN, shifted_config

NOMINAL = next(
    c.sim for c in expand_grid(grid('q3_s1_families')) if c.sim_name == NOMINAL_SIM
)
IDENTITY = {
    'feedback_c': NOMINAL.feedback_c,
    'wavelength_scale': 1.0,
    'wavelength_independent': 0.0,
}


@pytest.mark.parametrize('knob', sorted(IDENTITY))
def test_zero_shift_is_the_nominal_set(knob, tmp_path):
    # Same config -> same frozen file -> the nominal score, bit for bit.
    cfg = shifted_config(NOMINAL, knob, IDENTITY[knob])
    assert cfg == NOMINAL
    assert frozen_set_path(cfg, 1000, 64, tmp_path) == frozen_set_path(
        NOMINAL, 1000, 64, tmp_path
    )


@pytest.mark.parametrize(
    ('knob', 'values'),
    [
        ('feedback_c', (0.05, 0.2, 0.5)),
        ('wavelength_scale', (0.999, 1.0, 1.0001, 1.01)),
        ('wavelength_independent', (0.0, 1e-3, -1e-3)),
        ('phase_offset', (0.0, 0.5, 1.0)),
    ],
)
def test_distinct_shifts_have_distinct_paths(knob, values, tmp_path):
    paths = {
        frozen_set_path(shifted_config(NOMINAL, knob, v), 1000, 64, tmp_path)
        for v in values
    }
    assert len(paths) == len(values)


def test_wavelength_scale_multiplies_every_channel():
    cfg = shifted_config(NOMINAL, 'wavelength_scale', 1.01)
    assert cfg.wavelengths_um == pytest.approx(
        tuple(1.01 * w for w in NOMINAL.wavelengths_um)
    )
    assert dataclasses.replace(cfg, wavelengths_um=NOMINAL.wavelengths_um) == NOMINAL


def test_independent_shift_scales_each_channel_by_its_own_factor():
    cfg = shifted_config(NOMINAL, 'wavelength_independent', 1e-3)
    ratios = [
        a / b for a, b in zip(cfg.wavelengths_um, NOMINAL.wavelengths_um, strict=True)
    ]
    assert ratios == pytest.approx([1 + 1e-3 * p for p in WAVELENGTH_PATTERN])


def test_independent_shift_rejects_other_channel_counts():
    two = dataclasses.replace(NOMINAL, wavelengths_um=NOMINAL.wavelengths_um[:2])
    with pytest.raises(ValueError, match='needs 3 channels'):
        shifted_config(two, 'wavelength_independent', 1e-3)
