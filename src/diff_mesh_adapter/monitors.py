from __future__ import annotations

import torch

from diff_mesh_adapter.geometry import cell_centroids

Tensor = torch.Tensor


def advecting_gaussian_solution(points: Tensor, *, time: float, center0: tuple[float, float],
                                velocity: tuple[float, float], sigma: float) -> Tensor:
    center = _moving_center(points, time=time, center0=center0, velocity=velocity)
    dist2 = ((points - center) ** 2).sum(dim=1)
    return torch.exp(-dist2 / (2.0 * sigma ** 2))


def advecting_gaussian_monitor(
        points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        time: float = 0.35,
        center0: tuple[float, float] = (0.22, 0.42),
        velocity: tuple[float, float] = (0.9, 0.22),
        sigma: float = 0.085,
        alpha: float = 28.0,
) -> Tensor:
    centroids = cell_centroids(points, cell_blocks)
    solution = advecting_gaussian_solution(centroids, time=time, center0=center0, velocity=velocity, sigma=sigma)
    return 1.0 + alpha * solution


def advecting_gaussian_gradient_monitor(
        points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        time: float = 0.35,
        center0: tuple[float, float] = (0.22, 0.42),
        velocity: tuple[float, float] = (0.9, 0.22),
        sigma: float = 0.085,
        alpha: float = 0.24,
) -> Tensor:
    centroids = cell_centroids(points, cell_blocks)
    grad_norm = _gaussian_gradient_norm(centroids, time=time, center0=center0, velocity=velocity, sigma=sigma)
    return torch.sqrt(1.0 + alpha * grad_norm.square())


def time_integrated_advecting_gaussian_gradient_monitor(
        points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        time_start: float = 0.0,
        time_end: float = 0.85,
        samples: int = 7,
        center0: tuple[float, float] = (0.18, 0.30),
        velocity: tuple[float, float] = (0.65, 0.45),
        sigma: float = 0.075,
        alpha: float = 0.18,
) -> Tensor:
    if samples < 1:
        raise ValueError("samples must be positive")
    centroids = cell_centroids(points, cell_blocks)
    times = torch.linspace(time_start, time_end, samples, dtype=points.dtype, device=points.device)
    accumulated = torch.zeros(centroids.shape[0], dtype=points.dtype, device=points.device)
    for time in times:
        accumulated = accumulated + _gaussian_gradient_norm(
            centroids,
            time=time,
            center0=center0,
            velocity=velocity,
            sigma=sigma,
        ).square()
    mean_grad2 = accumulated / samples
    return torch.sqrt(1.0 + alpha * mean_grad2)


def advecting_smooth_front_solution(
        points: Tensor,
        *,
        time: float,
        front0: float,
        speed: float,
        normal: tuple[float, float],
        width: float,
) -> Tensor:
    normal_tensor = _unit_vector(points, normal)
    signed_distance = points @ normal_tensor - (front0 + speed * time)
    return 0.5 * (1.0 + torch.tanh(signed_distance / width))


def moving_gaussian_ridge_solution(
        points: Tensor,
        *,
        time: float,
        offset0: float,
        speed: float,
        normal: tuple[float, float],
        width: float,
) -> Tensor:
    signed_distance = _moving_line_signed_distance(points, time=time, offset0=offset0, speed=speed, normal=normal)
    return torch.exp(-signed_distance.square() / (2.0 * width ** 2))


def moving_gaussian_ridge_monitor(
        points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        time: float = 0.0,
        offset0: float = 0.16,
        speed: float = 0.82,
        normal: tuple[float, float] = (1.0, 0.18),
        width: float = 0.045,
        alpha: float = 16.0,
) -> Tensor:
    centroids = cell_centroids(points, cell_blocks)
    ridge = moving_gaussian_ridge_solution(
        centroids,
        time=time,
        offset0=offset0,
        speed=speed,
        normal=normal,
        width=width,
    )
    return 1.0 + alpha * ridge


def advecting_front_gradient_monitor(
        points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        time: float = 0.38,
        front0: float = 0.18,
        speed: float = 0.82,
        normal: tuple[float, float] = (1.0, 0.25),
        width: float = 0.035,
        alpha: float = 0.12,
) -> Tensor:
    centroids = cell_centroids(points, cell_blocks)
    normal_tensor = _unit_vector(points, normal)
    signed_distance = centroids @ normal_tensor - (front0 + speed * time)
    sech2 = 1.0 / torch.cosh(signed_distance / width).square()
    grad_norm = 0.5 * sech2 / width
    return torch.sqrt(1.0 + alpha * grad_norm.square())


def _moving_center(points: Tensor, *, time: float | Tensor, center0: tuple[float, float],
                   velocity: tuple[float, float]) -> Tensor:
    center = torch.tensor(center0, dtype=points.dtype, device=points.device)
    velocity_tensor = torch.tensor(velocity, dtype=points.dtype, device=points.device)
    return center + time * velocity_tensor


def _gaussian_gradient_norm(points: Tensor, *, time: float | Tensor, center0: tuple[float, float],
                            velocity: tuple[float, float], sigma: float) -> Tensor:
    center = _moving_center(points, time=time, center0=center0, velocity=velocity)
    offset = points - center
    solution = torch.exp(-offset.square().sum(dim=1) / (2.0 * sigma ** 2))
    return torch.linalg.norm(offset, dim=1) * solution / sigma ** 2


def _moving_line_signed_distance(
        points: Tensor,
        *,
        time: float | Tensor,
        offset0: float,
        speed: float,
        normal: tuple[float, float],
) -> Tensor:
    normal_tensor = _unit_vector(points, normal)
    return points @ normal_tensor - (offset0 + speed * time)


def _unit_vector(points: Tensor, vector: tuple[float, float]) -> Tensor:
    value = torch.tensor(vector, dtype=points.dtype, device=points.device)
    return value / torch.linalg.norm(value).clamp_min(torch.finfo(points.dtype).eps)
