from __future__ import annotations

from pathlib import Path


def generate_unit_square_gmsh(
    path: str | Path,
    *,
    boundary_size: float = 0.22,
    refined_size: float = 0.055,
    refined_point: tuple[float, float] = (0.28, 0.52),
) -> Path:
    """Generate a unit-square 2D Gmsh mesh with one embedded refinement point."""
    import gmsh

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("unit_square")

        geo = gmsh.model.geo
        p0 = geo.addPoint(0.0, 0.0, 0.0, boundary_size)
        p1 = geo.addPoint(1.0, 0.0, 0.0, boundary_size)
        p2 = geo.addPoint(1.0, 1.0, 0.0, boundary_size)
        p3 = geo.addPoint(0.0, 1.0, 0.0, boundary_size)
        pc = geo.addPoint(refined_point[0], refined_point[1], 0.0, refined_size)

        l0 = geo.addLine(p0, p1)
        l1 = geo.addLine(p1, p2)
        l2 = geo.addLine(p2, p3)
        l3 = geo.addLine(p3, p0)
        loop = geo.addCurveLoop([l0, l1, l2, l3])
        surface = geo.addPlaneSurface([loop])
        geo.synchronize()

        gmsh.model.mesh.embed(0, [pc], 2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()

    return output


def generate_unit_cube_gmsh(
    path: str | Path,
    *,
    boundary_size: float = 0.35,
    refined_size: float = 0.1,
) -> Path:
    """Generate a unit-cube 3D Gmsh mesh with one embedded refinement point."""
    import gmsh

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("unit_cube")

        occ = gmsh.model.occ
        volume = occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        pc = occ.addPoint(0.5, 0.5, 0.5, refined_size)
        occ.synchronize()

        boundary_points = sorted(
            {tag for dim, tag in gmsh.model.getBoundary([(3, volume)], recursive=True) if dim == 0}
        )
        gmsh.model.mesh.setSize([(0, tag) for tag in boundary_points], boundary_size)
        gmsh.model.mesh.setSize([(0, pc)], refined_size)
        gmsh.model.mesh.embed(0, [pc], 3, volume)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()

    return output


def generate_nonconvex_hole_gmsh(
    path: str | Path,
    *,
    boundary_size: float = 0.12,
    refined_size: float = 0.035,
    hole_size: float = 0.045,
) -> Path:
    """Generate an L-shaped nonconvex 2D mesh with a circular hole."""
    import gmsh

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("nonconvex_hole")

        geo = gmsh.model.geo
        outer_coords = [
            (0.0, 0.0),
            (1.4, 0.0),
            (1.4, 1.1),
            (0.85, 1.1),
            (0.85, 0.45),
            (0.0, 0.45),
        ]
        outer_points = [geo.addPoint(x, y, 0.0, boundary_size) for x, y in outer_coords]
        outer_lines = [
            geo.addLine(outer_points[i], outer_points[(i + 1) % len(outer_points)])
            for i in range(len(outer_points))
        ]
        outer_loop = geo.addCurveLoop(outer_lines)

        cx, cy, radius = 1.07, 0.25, 0.13
        center = geo.addPoint(cx, cy, 0.0, hole_size)
        hole_points = [
            geo.addPoint(cx + radius, cy, 0.0, hole_size),
            geo.addPoint(cx, cy + radius, 0.0, hole_size),
            geo.addPoint(cx - radius, cy, 0.0, hole_size),
            geo.addPoint(cx, cy - radius, 0.0, hole_size),
        ]
        arcs = [
            geo.addCircleArc(hole_points[i], center, hole_points[(i + 1) % len(hole_points)])
            for i in range(len(hole_points))
        ]
        hole_loop = geo.addCurveLoop(arcs)
        surface = geo.addPlaneSurface([outer_loop, hole_loop])

        refined_points = [
            geo.addPoint(0.62, 0.34, 0.0, refined_size),
            geo.addPoint(0.92, 0.78, 0.0, refined_size * 1.2),
        ]
        geo.synchronize()

        gmsh.model.mesh.embed(0, refined_points, 2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()

    return output
