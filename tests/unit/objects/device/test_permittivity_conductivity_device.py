"""Tests for PermittivityConductivityDevice.

The device carries two weights per voxel and interpolates permittivity and electric
conductivity independently along the same material ordering. These tests drive the real
place_objects + apply_params pipeline and check the resulting inverse-permittivity and
conductivity arrays.
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx import constants
from fdtdx.dispersion import DispersionModel, LorentzPole

LOW = fdtdx.Material(permittivity=2.0)
HIGH = fdtdx.Material(permittivity=12.0, electric_conductivity=5.0)


def _conductivity_scale(config):
    """Reproduce the scaling _init_arrays applies to physical conductivity."""
    return constants.c * config.time_step_duration / config.courant_number


def _build(materials, *, param_transforms=None, device_grid=(4, 4, 4)):
    config = fdtdx.SimulationConfig(
        time=1e-15, grid=fdtdx.UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=(8, 8, 8), material=fdtdx.Material(permittivity=1.0))
    device = fdtdx.PermittivityConductivityDevice(
        name="Device",
        materials=materials,
        param_transforms=[] if param_transforms is None else param_transforms,
        partial_grid_shape=device_grid,
        partial_voxel_grid_shape=(1, 1, 1),
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device], config=config, constraints=[device.place_at_center(volume)], key=key
    )
    return objects, arrays, params, config, key


def _apply(materials, perm_weight, cond_weight):
    objects, arrays, params, config, key = _build(materials)
    shape = params["Device"]["permittivity"].shape
    params["Device"] = {
        "permittivity": jnp.full(shape, perm_weight, dtype=jnp.float32),
        "conductivity": jnp.full(shape, cond_weight, dtype=jnp.float32),
    }
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    return objects["Device"], arrays, config


def _placed_device(materials, param_transforms, *, grid=(4, 4, 4), grid_slice=((0, 4), (0, 4), (0, 4))):
    config = fdtdx.SimulationConfig(
        time=1e-15, grid=fdtdx.UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None
    )
    device = fdtdx.PermittivityConductivityDevice(
        name="Device",
        materials=materials,
        param_transforms=param_transforms,
        partial_grid_shape=grid,
        partial_voxel_grid_shape=(1, 1, 1),
    )
    return device.place_on_grid(grid_slice_tuple=grid_slice, config=config, key=jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# Parameterization: two independent channels
# ---------------------------------------------------------------------------


def test_init_params_returns_both_channels():
    placed = _placed_device({"low": LOW, "high": HIGH}, [])
    params = placed.init_params(jax.random.PRNGKey(1))
    assert set(params.keys()) == {"permittivity", "conductivity"}
    assert params["permittivity"].shape == (4, 4, 4)
    assert params["conductivity"].shape == (4, 4, 4)
    assert float(jnp.min(params["permittivity"])) >= 0.0
    assert float(jnp.max(params["conductivity"])) <= 1.0


def test_call_returns_both_channels():
    placed = _placed_device({"low": LOW, "high": HIGH}, [])
    out = placed(
        {
            "permittivity": jnp.full((4, 4, 4), 0.3, jnp.float32),
            "conductivity": jnp.full((4, 4, 4), 0.7, jnp.float32),
        }
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"permittivity", "conductivity"}
    assert jnp.allclose(out["permittivity"], 0.3)
    assert jnp.allclose(out["conductivity"], 0.7)


def test_transforms_apply_to_each_channel_independently():
    placed = _placed_device(
        {"low": LOW, "high": HIGH},
        [fdtdx.GaussianSmoothing2D(std_discrete=1)],
        grid=(4, 4, 1),
        grid_slice=((0, 4), (0, 4), (0, 1)),
    )
    params = placed.init_params(jax.random.PRNGKey(1))
    assert set(params.keys()) == {"permittivity", "conductivity"}
    out = placed(params)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"permittivity", "conductivity"}
    # Smoothing is per-channel: distinct inputs stay distinct outputs.
    assert not jnp.allclose(out["permittivity"], out["conductivity"])
    assert jnp.all(jnp.isfinite(out["permittivity"])) and jnp.all(jnp.isfinite(out["conductivity"]))


def test_call_rejects_wrong_channel_names():
    placed = _placed_device({"low": LOW, "high": HIGH}, [])
    with pytest.raises(Exception, match="should return exactly the arrays"):
        placed({"permittivity": jnp.zeros((4, 4, 4)), "loss": jnp.zeros((4, 4, 4))})


# ---------------------------------------------------------------------------
# Materialization: permittivity / conductivity decoupling
# ---------------------------------------------------------------------------


def test_permittivity_and_conductivity_decouple():
    device, arrays, config = _apply({"low": LOW, "high": HIGH}, 0.5, 0.25)
    region_ip = arrays.inv_permittivities[:, *device.grid_slice]
    region_sigma = arrays.electric_conductivity[:, *device.grid_slice]
    # permittivity weight 0.5 -> eps = 2 + 0.5 * (12 - 2) = 7
    assert jnp.allclose(region_ip, 1.0 / 7.0)
    # conductivity weight 0.25 -> sigma = 0 + 0.25 * (5 - 0) = 1.25
    assert jnp.allclose(region_sigma, 1.25 * _conductivity_scale(config))


def test_channels_reach_opposite_corners_independently():
    # High permittivity with no loss.
    dev_a, arr_a, _ = _apply({"low": LOW, "high": HIGH}, 1.0, 0.0)
    assert jnp.allclose(arr_a.inv_permittivities[:, *dev_a.grid_slice], 1.0 / 12.0)
    assert jnp.allclose(arr_a.electric_conductivity[:, *dev_a.grid_slice], 0.0)
    # Low permittivity with full loss — a combination neither material provides.
    dev_b, arr_b, cfg_b = _apply({"low": LOW, "high": HIGH}, 0.0, 1.0)
    assert jnp.allclose(arr_b.inv_permittivities[:, *dev_b.grid_slice], 1.0 / 2.0)
    assert jnp.allclose(arr_b.electric_conductivity[:, *dev_b.grid_slice], 5.0 * _conductivity_scale(cfg_b))


def test_conductivity_written_outside_device_region_is_untouched():
    device, arrays, _ = _apply({"low": LOW, "high": HIGH}, 0.5, 1.0)
    mask = jnp.zeros(arrays.electric_conductivity.shape[1:], dtype=bool).at[device.grid_slice].set(True)
    outside = arrays.electric_conductivity[:, ~mask]
    assert jnp.allclose(outside, 0.0)


def test_diagonally_anisotropic_permittivity():
    low = fdtdx.Material(permittivity=(2.0, 3.0, 4.0))
    high = fdtdx.Material(permittivity=(12.0, 10.0, 8.0), electric_conductivity=5.0)
    device, arrays, _ = _apply({"low": low, "high": high}, 0.5, 0.0)
    region_ip = arrays.inv_permittivities[:, *device.grid_slice]
    assert region_ip.shape[0] == 3
    assert jnp.allclose(region_ip[0], 1.0 / 7.0)  # 2 + 0.5 * (12 - 2)
    assert jnp.allclose(region_ip[1], 1.0 / 6.5)  # 3 + 0.5 * (10 - 3)
    assert jnp.allclose(region_ip[2], 1.0 / 6.0)  # 4 + 0.5 * (8 - 4)


def test_diagonally_anisotropic_conductivity():
    low = fdtdx.Material(permittivity=2.0)
    high = fdtdx.Material(permittivity=12.0, electric_conductivity=(2.0, 4.0, 6.0))
    device, arrays, config = _apply({"low": low, "high": high}, 0.0, 1.0)
    region_sigma = arrays.electric_conductivity[:, *device.grid_slice]
    scale = _conductivity_scale(config)
    assert region_sigma.shape[0] == 3
    assert jnp.allclose(region_sigma[0], 2.0 * scale)
    assert jnp.allclose(region_sigma[1], 4.0 * scale)
    assert jnp.allclose(region_sigma[2], 6.0 * scale)


def test_gradient_flows_through_both_channels():
    objects, arrays, params, config, key = _build({"low": LOW, "high": HIGH})
    shape = params["Device"]["permittivity"].shape

    def loss(p):
        cur = dict(params)
        cur["Device"] = p
        new_arrays, _, _ = fdtdx.apply_params(arrays, objects, cur, key)
        return jnp.sum(new_arrays.inv_permittivities) + jnp.sum(new_arrays.electric_conductivity)

    p0 = {
        "permittivity": jnp.full(shape, 0.5, jnp.float32),
        "conductivity": jnp.full(shape, 0.5, jnp.float32),
    }
    grad = jax.grad(loss)(p0)
    assert jnp.all(jnp.isfinite(grad["permittivity"])) and jnp.any(grad["permittivity"] != 0)
    assert jnp.all(jnp.isfinite(grad["conductivity"])) and jnp.any(grad["conductivity"] != 0)


def test_apply_is_idempotent():
    objects, arrays, params, config, key = _build({"low": LOW, "high": HIGH})
    once, objects_once, _ = fdtdx.apply_params(arrays, objects, params, key)
    twice, _, _ = fdtdx.apply_params(once, objects_once, params, key)
    assert jnp.allclose(once.inv_permittivities, twice.inv_permittivities)
    assert jnp.allclose(once.electric_conductivity, twice.electric_conductivity)


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


def test_warns_without_conductive_material_but_still_optimizes_permittivity():
    materials = {"low": LOW, "high": fdtdx.Material(permittivity=12.0)}  # no loss anywhere
    with pytest.warns(UserWarning, match="none of its materials"):
        device, arrays, _ = _apply(materials, 0.5, 0.5)
    # Nothing in the simulation is lossy, so no conductivity array is allocated at all.
    assert arrays.electric_conductivity is None
    # The permittivity channel still works: eps = 2 + 0.5 * (12 - 2) = 7
    assert jnp.allclose(arrays.inv_permittivities[:, *device.grid_slice], 1.0 / 7.0)


def test_requires_two_materials():
    with pytest.raises(Exception, match="exactly two materials"):
        _placed_device({"high": HIGH}, [])


def test_rejects_discrete_output():
    with pytest.raises(Exception, match="only supports continuous parameters"):
        _placed_device({"low": LOW, "high": HIGH}, [fdtdx.ClosestIndex()])


def test_dispersive_materials_not_supported():
    disp = DispersionModel(poles=(LorentzPole(resonance_frequency=2e14, damping=1e13, delta_epsilon=2.0),))
    high = fdtdx.Material(permittivity=12.0, electric_conductivity=5.0, dispersion=disp)
    with pytest.raises(NotImplementedError, match="does not support dispersive"):
        _apply({"low": LOW, "high": high}, 0.5, 0.5)


# ---------------------------------------------------------------------------
# The single-channel Device interface must keep working unchanged
# ---------------------------------------------------------------------------


def test_standard_device_still_returns_a_bare_array():
    config = fdtdx.SimulationConfig(
        time=1e-15, grid=fdtdx.UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None
    )
    device = fdtdx.Device(
        name="Device",
        materials={"low": LOW, "high": fdtdx.Material(permittivity=12.0)},
        param_transforms=[],
        partial_grid_shape=(4, 4, 4),
        partial_voxel_grid_shape=(1, 1, 1),
    )
    placed = device.place_on_grid(
        grid_slice_tuple=((0, 4), (0, 4), (0, 4)), config=config, key=jax.random.PRNGKey(0)
    )
    assert placed.output_names == ("params",)
    params = placed.init_params(jax.random.PRNGKey(1))
    assert isinstance(params, jax.Array)
    out = placed(params)
    assert isinstance(out, jax.Array)
    assert out.shape == (4, 4, 4)
