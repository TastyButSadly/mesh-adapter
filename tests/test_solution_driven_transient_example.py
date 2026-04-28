from examples.applications.solution_driven_transient_adaptation_2d import run_case


def test_solution_driven_transient_example_writes_paraview_series(tmp_path):
    paths = run_case(tmp_path, mesh_size=0.16, steps_per_time=4, time_count=3)

    pvd = paths["pvd"]
    frames = paths["frames"]

    assert pvd.exists()
    assert pvd.stat().st_size > 0
    assert len(frames) == 3
    for frame in frames:
        assert frame.exists()
        assert frame.stat().st_size > 0
