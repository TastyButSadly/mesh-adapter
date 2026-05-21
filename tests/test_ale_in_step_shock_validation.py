from examples.applications.ale_in_step_shock_validation import run_burgers_shock, run_gcl_check


def test_direct_ale_step_preserves_constant_state_under_mesh_motion():
    metrics = run_gcl_check()

    assert metrics["ale_max_abs_error"] < 1e-8
    assert metrics["ale_mass_error"] < 1e-10
    assert metrics["naive_without_mesh_flux_max_abs_error"] > 1.0


def test_adaptive_every_step_burgers_shock_tracks_speed_and_sharpens_front():
    fixed = run_burgers_shock(cells=80, final_time=0.25, adaptive=False)
    adaptive = run_burgers_shock(cells=80, final_time=0.25, adaptive=True)

    assert abs(adaptive.metrics.shock_speed_error) < 5e-3
    assert abs(adaptive.metrics.mass_error) < 1e-12
    assert adaptive.metrics.bounds_violation == 0.0
    assert adaptive.metrics.front_width_10_90 < 0.5 * fixed.metrics.front_width_10_90
    assert adaptive.metrics.l1_error_exact < 0.5 * fixed.metrics.l1_error_exact
    assert adaptive.metrics.near_shock_dx_ratio < 0.5
