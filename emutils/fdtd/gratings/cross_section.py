"""Decomposition of the grating coupler cross-section into non-overlapping rectangles."""
from dataclasses import dataclass
from typing import List

import numpy as np

from .geometry import GratingGeometry, _is_air


@dataclass(frozen=True)
class Rect:
    name: str
    material: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    in_grating: bool = False


def profile_intervals(geom: GratingGeometry, buffer: float, include_input: bool = True):
    """
    Splits the propagation axis into intervals (label, x_min, x_max, surface level).
    Grating starts at x=0, input waveguide (if included) spans x < 0, tail of length
    `buffer` follows the last segment.
    """
    y_top, _, surfaces = geom.section_surfaces(buffer)

    intervals = [('input', -buffer - geom.input_wg_length, 0.0, y_top)] if include_input else []
    x = 0.0
    for i in range(geom.n_segments):
        for j, (ff, surface) in enumerate(zip(geom.fill_factor, surfaces)):
            if ff == 0:
                continue
            intervals.append((f'seg{i}_sec{j}', x, x + ff * geom.pitch, surface))
            x += ff * geom.pitch
    intervals.append(('tail', x, x + buffer, y_top))
    return intervals


def layer_runs(geom: GratingGeometry, intervals, buffer: float) -> List[Rect]:
    """
    Builds non-overlapping rectangles of the stack cross-section over given `intervals`
    (see `profile_intervals`). Consecutive intervals with equal layer top are merged.
    """
    y_top, fill, _ = geom.section_surfaces(buffer)

    rects = []
    for name, (y_min, y_max) in geom.layer_bounds(buffer).items():
        material = geom.layers_stack[name][0]
        if _is_air(material):
            continue
        above_etch = y_min >= y_top
        # merge consecutive intervals with equal clipped layer top
        runs = []
        for label, x0, x1, surface in intervals:
            top = y_max if above_etch else min(y_max, surface)
            if runs and np.isclose(runs[-1][3], top, rtol=0, atol=1e-15):
                runs[-1][2] = x1
                runs[-1][4] = label
            else:
                runs.append([label, x0, x1, top, label])
        if len(runs) == 1:
            _, x0, x1, top, _ = runs[0]
            rects.append(Rect(name, material, x0, x1, y_min, top))
            continue
        for first, x0, x1, top, last in runs:
            if top <= y_min:
                continue
            label = first if first == last else f'{first}-{last}'
            rects.append(Rect(f'{name}_{label}', material, x0, x1, y_min, top, in_grating=True))

    if not _is_air(fill):
        for label, x0, x1, surface in intervals:
            if surface < y_top:
                rects.append(Rect(f'fill_{label}', fill, x0, x1, surface, y_top, in_grating=True))

    return rects


def build_rects(geom: GratingGeometry, buffer: float) -> List[Rect]:
    """
    Builds the list of non-overlapping rectangles representing the whole grating coupler
    cross-section. Grating starts at x=0, input waveguide spans x < 0.
    """
    return layer_runs(geom, profile_intervals(geom, buffer), buffer)
