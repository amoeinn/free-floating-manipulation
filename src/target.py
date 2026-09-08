"""A parametric client satellite, defined here rather than loaded from a file.

The premise of phase 4 is that the servicer has no drawings of its client, so
the target's geometry must not leak into the reconstruction. It is defined in
one place, used only to make images, and never read by anything that fits.

The shape is chosen to make a reconstruction falsifiable by eye. A bus alone
would be a box that almost any blob of Gaussians could fake. Panels are thin,
so they test whether the fit resolves a surface rather than a cloud; the dish
is curved and self shadowing; the boom is a thin cylinder near the resolution
limit, which is where a small fit is expected to fail first and is therefore
worth including rather than leaving out.
"""

from __future__ import annotations

import numpy as np

from .raytrace import Box, Cylinder, Scene


# Metres. A small servicing client, roughly ESA e.Deorbit scale.
BUS_HALF = np.array([0.55, 0.45, 0.65])
PANEL_HALF = np.array([1.30, 0.012, 0.55])
PANEL_OFFSET = 1.55
DISH_RADIUS = 0.38
DISH_THICKNESS = 0.09
BOOM_RADIUS = 0.035
BOOM_LENGTH = 0.95

ALBEDO_BUS = np.array([0.62, 0.60, 0.55])          # foil and panelling
ALBEDO_PANEL = np.array([0.10, 0.14, 0.32])        # dark blue cells
ALBEDO_DISH = np.array([0.86, 0.85, 0.82])         # white antenna
ALBEDO_BOOM = np.array([0.45, 0.44, 0.42])


def client_satellite() -> list:
    """The primitives making up the target, in the target's own frame."""
    return [
        Box(position=[0.0, 0.0, 0.0], rpy=[0, 0, 0], albedo=ALBEDO_BUS,
            name="bus", half_extents=BUS_HALF),
        Box(position=[0.0, +PANEL_OFFSET, 0.0], rpy=[0, 0, 0], albedo=ALBEDO_PANEL,
            name="panel_plus_y", half_extents=PANEL_HALF),
        Box(position=[0.0, -PANEL_OFFSET, 0.0], rpy=[0, 0, 0], albedo=ALBEDO_PANEL,
            name="panel_minus_y", half_extents=PANEL_HALF),
        Cylinder(position=[0.0, 0.0, BUS_HALF[2] + DISH_THICKNESS / 2], rpy=[0, 0, 0],
                 albedo=ALBEDO_DISH, name="dish",
                 radius=DISH_RADIUS, half_height=DISH_THICKNESS / 2),
        Cylinder(position=[BUS_HALF[0] + BOOM_LENGTH / 2, 0.0, -0.25],
                 rpy=[0.0, np.pi / 2, 0.0], albedo=ALBEDO_BOOM, name="boom",
                 radius=BOOM_RADIUS, half_height=BOOM_LENGTH / 2),
    ]


def client_scene(sun_direction=(-0.42, -0.60, -0.68), intensity: float = 1.0) -> Scene:
    """The target under a single hard light, which is what orbit provides.

    The default sun direction is off axis on all three axes so that no face is
    lit head on and no two faces receive the same irradiance. A light along a
    principal axis would make the bus faces indistinguishable and would hide a
    whole class of normal errors.
    """
    return Scene(primitives=client_satellite(),
                 light_direction=np.asarray(sun_direction, float),
                 light_intensity=intensity)


def bounding_radius() -> float:
    """Radius of a sphere at the target origin containing every primitive.

    Used to place cameras and, later, to initialise a fit without telling it
    anything about the shape beyond its extent.
    """
    corners = []
    for prim in client_satellite():
        if isinstance(prim, Box):
            h = prim.half_extents
        else:
            h = np.array([prim.radius, prim.radius, prim.half_height])
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        local = signs * h
        corners.append(local @ prim._R.T + prim.position)
    return float(np.linalg.norm(np.concatenate(corners), axis=1).max())
