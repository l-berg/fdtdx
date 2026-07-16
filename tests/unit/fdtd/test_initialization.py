import warnings
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.constants import MAX_SIMULATION_VOLUME_CELLS
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.dispersion import DispersionModel, LorentzPole
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.fdtd.initialization import (
    _apply_grid_coordinate_constraint,
    _apply_position_constraint,
    _apply_real_coordinate_constraint,
    _apply_size_constraint,
    _apply_size_extension_constraint,
    _check_objects_names_from_constraints,
    _handle_unresolved_objects,
    _init_arrays,
    _resolve_grid_from_volume,
    _resolve_static_shapes,
    _resolve_volume_name,
    _update_grid_shapes_from_slices,
    _update_grid_slices_from_shapes,
    _warn_if_simulation_volume_too_large,
    apply_params,
    place_objects,
    resolve_object_constraints,
)
from fdtdx.materials import Material, compute_allowed_dispersive_coefficients
from fdtdx.objects.device.device import Device, EtchedDevice
from fdtdx.objects.device.parameters.discretization import ClosestIndex
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.static_material.static import (
    SimulationVolume,
    UniformMaterialObject,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config():
    return SimulationConfig(grid=UniformGrid(spacing=1.0), time=100e-15)


@pytest.fixture
def simple_volume():
    return SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))


@pytest.fixture
def simple_material():
    return Material(
        permittivity=(2.0, 2.0, 2.0),
        permeability=(1.0, 1.0, 1.0),
        electric_conductivity=(0.0, 0.0, 0.0),
        magnetic_conductivity=(0.0, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# resolve_object_constraints - error path tests
# ---------------------------------------------------------------------------


def test_resolve_constraints_with_duplicate_object_names(simple_config, simple_volume):
    volume2 = SimulationVolume(name="volume", partial_grid_shape=(50, 50, 50))
    with pytest.raises(Exception, match="Duplicate object names"):
        resolve_object_constraints([simple_volume, volume2], [], simple_config)


def test_resolve_constraints_unknown_object_in_constraint(simple_config, simple_volume):
    constraint = GridCoordinateConstraint(object="nonexistent_object", axes=[0], sides=["-"], coordinates=[10])
    with pytest.raises(ValueError, match="Unknown object name"):
        resolve_object_constraints([simple_volume], [constraint], simple_config)


def test_resolve_constraints_no_simulation_volume(simple_config, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    with pytest.raises(ValueError, match="No SimulationVolume"):
        resolve_object_constraints([obj], [], simple_config)


def test_resolve_constraints_multiple_volumes(simple_config):
    volume1 = SimulationVolume(name="volume1", partial_grid_shape=(100, 100, 100))
    volume2 = SimulationVolume(name="volume2", partial_grid_shape=(50, 50, 50))
    with pytest.raises(ValueError, match="Multiple SimulationVolume"):
        resolve_object_constraints([volume1, volume2], [], simple_config)


def test_resolve_constraints_conflicting_grid_coordinates(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    c1 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])
    c2 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    _resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [c1, c2], simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_conflicting_real_coordinates(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    c1 = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10.0])
    c2 = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20.0])
    _resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [c1, c2], simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_inconsistent_size_and_position(simple_config, simple_volume, simple_material):
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    size_constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[0],
        offsets=[None],
    )
    # Position constraint: obj1 is 5 cells distant from the volume lower bound, at the center
    pos_constraint = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],
        grid_margins=[5],
        margins=[None],
    )
    # Grid constraint: obj1 is at the the volume's lower edge (physical coordinate -5)
    real_constraint = RealCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[-5.0],  # Center of the volume (origin)
    )
    _resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj], [size_constraint, pos_constraint, real_constraint], config
    )
    assert errors["obj1"] is not None


def test_resolve_constraints_with_real_margins(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    size_constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0, 1, 2],
        other_axes=[0, 1, 2],
        proportions=[0.1, 0.1, 0.1],
        grid_offsets=[0, 0, 0],
        offsets=[None, None, None],
    )
    pos_constraint = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],
        grid_margins=[None],
        margins=[5.0],
    )
    resolved_slices, _errors = resolve_object_constraints(
        [simple_volume, obj], [size_constraint, pos_constraint], simple_config
    )
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_both_margins(simple_volume, simple_material):
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    constraint = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],
        grid_margins=[2],
        margins=[1.0],
    )
    resolved_slices, _errors = resolve_object_constraints([simple_volume, obj], [constraint], config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_real_offset_in_size(simple_volume, simple_material):
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[None],
        offsets=[2.0],
    )
    resolved_slices, _errors = resolve_object_constraints([simple_volume, obj], [constraint], config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_grid_offset_in_size(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[10],
        offsets=[None],
    )
    resolved_slices, _errors = resolve_object_constraints([simple_volume, obj], [constraint], simple_config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_size_extension_with_real_offset(simple_volume, simple_material):
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", material=simple_material)
    pos_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    ext_constraint = SizeExtensionConstraint(
        object="obj2",
        other_object="obj1",
        axis=0,
        direction="+",
        other_position=1.0,
        grid_offset=None,
        offset=2.0,
    )
    resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj1, obj2], [pos_constraint, ext_constraint], config
    )
    assert errors["obj2"] is None
    assert resolved_slices["obj2"][0][1] == 34


def test_resolve_constraints_size_extension_with_grid_offset(simple_config, simple_volume, simple_material):
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", material=simple_material)
    pos_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    ext_constraint = SizeExtensionConstraint(
        object="obj2",
        other_object="obj1",
        axis=0,
        direction="-",
        other_position=-1.0,
        grid_offset=5,
        offset=None,
    )
    resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj1, obj2], [pos_constraint, ext_constraint], simple_config
    )
    assert errors["obj2"] is None
    assert resolved_slices["obj2"][0][0] == 25


def test_resolve_constraints_size_extension_to_volume_boundary(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20]),
        GridCoordinateConstraint(object="obj1", axes=[1], sides=["-"], coordinates=[20]),
        SizeExtensionConstraint(
            object="obj1",
            other_object=None,
            axis=2,
            direction="-",
            other_position=0.0,
            grid_offset=None,
            offset=None,
        ),
    ]
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], constraints, simple_config)
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][2][0] == 0


def test_resolve_constraints_with_partial_real_shape(simple_material):
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    obj = UniformMaterialObject(name="obj1", partial_real_shape=(5.0, 5.0, 5.0), material=simple_material)
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[0, 0, 0])
    ]
    resolved_slices, _errors = resolve_object_constraints([volume, obj], constraints, config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices
    if resolved_slices["obj1"][0][0] is not None:
        shape = resolved_slices["obj1"][0][1] - resolved_slices["obj1"][0][0]
        assert shape == 10


def test_nonuniform_partial_real_shape_covers_metric_size(simple_material):
    """Real object sizes use grid edges and cover the requested metric length."""
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
        y_edges=jnp.asarray([0.0, 2.0, 5.0]),
        z_edges=jnp.asarray([0.0, 1.5, 4.0]),
    )
    config = SimulationConfig(grid=grid, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=grid.shape)
    obj = UniformMaterialObject(name="obj1", partial_real_shape=(2.1, 2.1, 2.1), material=simple_material)
    constraints = [
        RealCoordinateConstraint(object="obj1", axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0.0, 0.0, 0.0))
    ]

    resolved_slices, errors = resolve_object_constraints([volume, obj], constraints, config)

    assert errors["obj1"] is None
    assert resolved_slices["obj1"] == ((0, 2), (0, 2), (0, 2))


def test_nonuniform_partial_real_position_uses_physical_interval_center(simple_material):
    """Center placement chooses the grid interval with closest physical center."""
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
        y_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
        z_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
    )
    config = SimulationConfig(grid=grid, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=grid.shape)
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(2, 2, 2),
        partial_real_position=(0.6, 0.6, 0.6),
        material=simple_material,
    )

    resolved_slices, errors = resolve_object_constraints([volume, obj], [], config)

    assert errors["obj1"] is None
    assert resolved_slices["obj1"] == ((1, 3), (1, 3), (1, 3))


def test_nonuniform_real_coordinate_constraint_snaps_to_nearest_edge(simple_material):
    """Real coordinate constraints use physical edge coordinates on stretched grids."""
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
        y_edges=jnp.asarray([0.0, 1.0]),
        z_edges=jnp.asarray([0.0, 1.0]),
    )
    config = SimulationConfig(grid=grid, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=grid.shape)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(1, 1, 1), material=simple_material)
    constraint = RealCoordinateConstraint(object="obj1", axes=(0,), sides=("-",), coordinates=(2.7,))

    resolved_slices, errors = resolve_object_constraints([volume, obj], [constraint], config)

    assert errors["obj1"] is None
    assert resolved_slices["obj1"][0] == (2, 3)


def test_nonuniform_grid_coordinate_constraint_is_rejected(simple_material):
    """Index-space placement coordinates are not allowed on non-uniform grids."""
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
        y_edges=jnp.asarray([0.0, 1.0, 2.0, 3.0]),
        z_edges=jnp.asarray([0.0, 1.0, 2.0, 3.0]),
    )
    config = SimulationConfig(grid=grid, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=grid.shape)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(1, 1, 1), material=simple_material)
    constraint = GridCoordinateConstraint(object="obj1", axes=(0,), sides=("-",), coordinates=(1,))

    _resolved_slices, errors = resolve_object_constraints([volume, obj], [constraint], config)

    assert "not supported on non-uniform grids" in errors["obj1"]


def test_nonuniform_nonzero_grid_margin_is_rejected(simple_material):
    """Grid margins are index-space distances and must be expressed in metres."""
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0, 10.0]),
        y_edges=jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        z_edges=jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    config = SimulationConfig(grid=grid, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=grid.shape)
    parent = UniformMaterialObject(name="parent", partial_grid_shape=(1, 1, 1), material=simple_material)
    child = UniformMaterialObject(name="child", partial_grid_shape=(1, 1, 1), material=simple_material)
    constraints = [
        RealCoordinateConstraint(object="parent", axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0.0, 0.0, 0.0)),
        child.face_to_face_positive_direction(parent, axes=(0,), grid_margins=(1,)),
    ]

    _resolved_slices, errors = resolve_object_constraints([volume, parent, child], constraints, config)

    assert "grid_margins" in errors["child"]


def test_nonuniform_size_constraint_uses_physical_proportion(simple_material):
    """SizeConstraint proportions resolve using metric extents on stretched grids.

    Grid x has widths [1, 2, 3] so three cells span 6 m.  A 50 % proportion
    targets 3 m, which covers the first two cells (edges 0→1→3).
    """
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),  # widths 1, 2, 3 → total 6 m
        y_edges=jnp.asarray([0.0, 1.0]),
        z_edges=jnp.asarray([0.0, 1.0]),
    )
    config = SimulationConfig(grid=grid, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=grid.shape)
    parent = UniformMaterialObject(name="parent", partial_grid_shape=(3, 1, 1), material=simple_material)
    child = UniformMaterialObject(name="child", material=simple_material)
    constraints = [
        RealCoordinateConstraint(object="parent", axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0.0, 0.0, 0.0)),
        RealCoordinateConstraint(object="child", axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0.0, 0.0, 0.0)),
        SizeConstraint(
            object="child",
            other_object="parent",
            axes=[0],
            other_axes=[0],
            proportions=[0.5],
            grid_offsets=[None],
            offsets=[None],
        ),
    ]

    resolved_slices, errors = resolve_object_constraints([volume, parent, child], constraints, config)

    assert errors["parent"] is None
    assert errors["child"] is None
    assert resolved_slices["parent"][0] == (0, 3)
    # 50 % of 6 m = 3 m; upper-snap from lower edge gives 2 cells (0→1→3)
    child_size = resolved_slices["child"][0][1] - resolved_slices["child"][0][0]
    assert child_size == 2


def test_resolve_constraints_extend_to_infinity(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [constraint], simple_config)
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


# ---------------------------------------------------------------------------
# _resolve_volume_name - direct unit tests
# ---------------------------------------------------------------------------


def test_resolve_volume_name_success(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {simple_volume.name: simple_volume, obj.name: obj}
    assert _resolve_volume_name(obj_map) == "volume"


def test_resolve_volume_name_no_volume(simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    with pytest.raises(ValueError, match="No SimulationVolume"):
        _resolve_volume_name({"obj1": obj})


def test_resolve_volume_name_multiple_volumes():
    v1 = SimulationVolume(name="v1", partial_grid_shape=(10, 10, 10))
    v2 = SimulationVolume(name="v2", partial_grid_shape=(10, 10, 10))
    with pytest.raises(ValueError, match="Multiple SimulationVolume"):
        _resolve_volume_name({"v1": v1, "v2": v2})


# ---------------------------------------------------------------------------
# _check_objects_names_from_constraints - direct unit tests
# ---------------------------------------------------------------------------


def test_check_objects_names_no_constraints():
    result = _check_objects_names_from_constraints([], ["volume", "obj1"])
    assert result == []


def test_check_objects_names_valid_names():
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])
    result = _check_objects_names_from_constraints([c], ["obj1", "volume"])
    assert "obj1" in result


def test_check_objects_names_unknown_name():
    c = GridCoordinateConstraint(object="missing", axes=[0], sides=["-"], coordinates=[10])
    with pytest.raises(ValueError, match="Unknown object name"):
        _check_objects_names_from_constraints([c], ["obj1", "volume"])


def test_check_objects_names_other_object_unknown():
    c = SizeConstraint(
        object="obj1",
        other_object="missing",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[0],
        offsets=[None],
    )
    with pytest.raises(ValueError, match="Unknown object name"):
        _check_objects_names_from_constraints([c], ["obj1", "volume"])


# ---------------------------------------------------------------------------
# _resolve_static_shapes - direct unit tests
# ---------------------------------------------------------------------------


def test_resolve_static_shapes_grid_shape(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 20, 30), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    result = _resolve_static_shapes(obj_map, shape_dict, simple_config)
    assert result["obj1"] == [10, 20, 30]


def test_resolve_static_shapes_real_shape(simple_material):
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    config = _resolve_grid_from_volume([volume], config)
    obj = UniformMaterialObject(name="obj1", partial_real_shape=(5.0, 10.0, 15.0), material=simple_material)
    obj_map = {"volume": volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    result = _resolve_static_shapes(obj_map, shape_dict, config)
    # 5.0 / 0.5 = 10, 10.0 / 0.5 = 20, 15.0 / 0.5 = 30
    assert result["obj1"] == [10, 20, 30]


def test_resolve_static_shapes_no_shape(simple_config, simple_material):
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    result = _resolve_static_shapes(obj_map, shape_dict, simple_config)
    # volume has grid shape, obj1 has nothing
    assert result["volume"] == [100, 100, 100]
    assert result["obj1"] == [None, None, None]


# ---------------------------------------------------------------------------
# _apply_grid_coordinate_constraint - direct unit tests
# ---------------------------------------------------------------------------


def test_apply_grid_coordinate_constraint_sets_lower(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15])
    resolved, new_slices = _apply_grid_coordinate_constraint(c, obj_map, slice_dict)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 15


def test_apply_grid_coordinate_constraint_sets_upper(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["+"], coordinates=[50])
    resolved, new_slices = _apply_grid_coordinate_constraint(c, obj_map, slice_dict)
    assert resolved is True
    assert new_slices["obj1"][0][1] == 50


def test_apply_grid_coordinate_constraint_consistent_no_change(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[15, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15])
    resolved, _new_slices = _apply_grid_coordinate_constraint(c, obj_map, slice_dict)
    assert resolved is False


def test_apply_grid_coordinate_constraint_conflicting_raises(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[15, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_grid_coordinate_constraint(c, obj_map, slice_dict)


# ---------------------------------------------------------------------------
# _apply_real_coordinate_constraint - direct unit tests
# ---------------------------------------------------------------------------


def test_apply_real_coordinate_constraint_converts_to_grid(simple_config, simple_volume, simple_material):
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    # With resolution=1.0, coordinate 15.0 -> grid 65
    c = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15.0])
    resolved, new_slices = _apply_real_coordinate_constraint(c, obj_map, slice_dict, config)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 65


def test_apply_real_coordinate_constraint_sub_resolution(simple_material):
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    config = _resolve_grid_from_volume([volume], config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    # resolution=0.5, coordinate 5.0 -> grid  5 /(0.5) + 50 = 60
    c = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[5.0])
    resolved, new_slices = _apply_real_coordinate_constraint(c, obj_map, slice_dict, config)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 60


def test_apply_real_coordinate_constraint_conflicting_raises(simple_config, simple_volume, simple_material):
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[15, None], [None, None], [None, None]]}
    c = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20.0])
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_real_coordinate_constraint(c, obj_map, slice_dict, config)


# ---------------------------------------------------------------------------
# _apply_size_extension_constraint - error path
# ---------------------------------------------------------------------------


def test_apply_size_extension_constraint_other_not_placed_yet(simple_config, simple_volume, simple_material):
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj1, "obj2": obj2}
    # obj1 not yet placed (None boundaries)
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, None], [None, None], [None, None]],
        "obj2": [[None, None], [None, None], [None, None]],
    }
    c = SizeExtensionConstraint(
        object="obj2",
        other_object="obj1",
        axis=0,
        direction="+",
        other_position=1.0,
        grid_offset=None,
        offset=None,
    )
    resolved, _ = _apply_size_extension_constraint(c, obj_map, config, slice_dict, "volume")
    assert resolved is False


# ---------------------------------------------------------------------------
# _handle_unresolved_objects - direct unit tests
# ---------------------------------------------------------------------------


def test_handle_unresolved_objects_marks_error(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    result = _handle_unresolved_objects(obj_map, slice_dict, errors)
    assert result["obj1"] is not None
    assert result["volume"] is None


def test_handle_unresolved_objects_no_errors_when_resolved(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, 15], [5, 15], [5, 15]],
    }
    errors = {"volume": None, "obj1": None}
    result = _handle_unresolved_objects(obj_map, slice_dict, errors)
    assert result["obj1"] is None
    assert result["volume"] is None


# ---------------------------------------------------------------------------
# apply_params - device materialization (real devices)
# ---------------------------------------------------------------------------
#
# The material-writing logic (continuous vs discrete, isotropic vs anisotropic,
# plain vs etched, dispersive coefficients) used to live inline in apply_params
# and was exercised here with heavily mocked devices. It now lives on the Device
# (Device.apply_to_arrays / EtchedDevice), so these tests drive real devices
# through the full place_objects + apply_params pipeline and check the resulting
# inverse-permittivity / dispersive-coefficient arrays.


def _device(cls=Device, *, materials, param_transforms=None, device_grid=(4, 4, 4)):
    return cls(
        name="Device",
        materials=materials,
        param_transforms=param_transforms or [],
        partial_grid_shape=device_grid,
        partial_voxel_grid_shape=(1, 1, 1),
    )


def _apply_device_params(device, volume_material, params_value, *, grid_shape=(8, 8, 8)):
    """Place volume + device, set constant params, apply, return (placed_device, arrays, config)."""
    config = SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=50e-9),
        backend="cpu",
        gradient_config=None,
    )
    volume = SimulationVolume(partial_grid_shape=grid_shape, material=volume_material)
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects(
        object_list=[volume, device],
        config=config,
        constraints=[device.place_at_center(volume)],
        key=key,
    )
    params["Device"] = jnp.full(jnp.asarray(params["Device"]).shape, params_value, dtype=jnp.float32)
    arrays, objects, _ = apply_params(arrays, objects, params, key)
    return objects["Device"], arrays, config


def test_apply_params_continuous_isotropic():
    materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
    placed, arrays, _ = _apply_device_params(_device(materials=materials), Material(permittivity=1.0), 0.5)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert region.shape[0] == 1
    # 50% interpolation between eps 1.0 and 12.25 -> 6.625
    assert jnp.allclose(region, 1.0 / 6.625)
    # background (air) outside the device is untouched
    assert jnp.allclose(arrays.inv_permittivities[0, 0, 0, 0], 1.0)


def test_apply_params_continuous_diagonally_anisotropic():
    materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=(12.25, 10.0, 8.0))}
    placed, arrays, _ = _apply_device_params(_device(materials=materials), Material(permittivity=1.0), 0.5)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert region.shape[0] == 3
    # per-axis 50% between 1.0 and (12.25, 10.0, 8.0) -> (6.625, 5.5, 4.5)
    assert jnp.allclose(region[0], 1.0 / 6.625)
    assert jnp.allclose(region[1], 1.0 / 5.5)
    assert jnp.allclose(region[2], 1.0 / 4.5)


def test_apply_params_continuous_fully_anisotropic():
    tensor_b = (4.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 4.0)
    materials = {"a": Material(permittivity=1.0), "b": Material(permittivity=tensor_b)}
    placed, arrays, _ = _apply_device_params(_device(materials=materials), Material(permittivity=1.0), 0.5)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert region.shape[0] == 9
    identity = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    interp = 0.5 * identity + 0.5 * np.array(tensor_b)
    expected = np.linalg.inv(interp.reshape(3, 3)).flatten()
    assert np.allclose(np.array(region[:, 0, 0, 0]), expected, atol=1e-5)


def test_apply_params_discrete_isotropic_selects_material():
    materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
    device = _device(materials=materials, param_transforms=[ClosestIndex()])
    # params 0.7 rounds to index 1 -> si
    placed, arrays, _ = _apply_device_params(device, Material(permittivity=1.0), 0.7)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert jnp.allclose(region, 1.0 / 12.25)


def test_apply_params_discrete_fully_anisotropic_selects_material():
    tensor_b = (4.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 4.0)
    materials = {"a": Material(permittivity=1.0), "b": Material(permittivity=tensor_b)}
    device = _device(materials=materials, param_transforms=[ClosestIndex()])
    # params 0.9 rounds to index 1 -> b
    placed, arrays, _ = _apply_device_params(device, Material(permittivity=1.0), 0.9)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert region.shape[0] == 9
    expected = np.linalg.inv(np.array(tensor_b).reshape(3, 3)).flatten()
    assert np.allclose(np.array(region[:, 0, 0, 0]), expected, atol=1e-5)


def test_apply_params_etched_continuous_isotropic():
    # background eps 2.0, etch material eps 4.0, 50% mix -> eps 3.0 -> inv 1/3
    device = _device(cls=EtchedDevice, materials={"etch": Material(permittivity=4.0)})
    placed, arrays, _ = _apply_device_params(device, Material(permittivity=2.0), 0.5)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert jnp.allclose(region, 1.0 / 3.0)


def test_apply_params_etched_continuous_diagonally_anisotropic():
    # background diag(2,2,2), etch diag(1, 2, 4); 50% mix -> (1.5, 2.0, 3.0)
    device = _device(cls=EtchedDevice, materials={"etch": Material(permittivity=(1.0, 2.0, 4.0))})
    placed, arrays, _ = _apply_device_params(device, Material(permittivity=2.0), 0.5)
    region = arrays.inv_permittivities[:, *placed.grid_slice]
    assert region.shape[0] == 3
    assert jnp.allclose(region[0], 1.0 / 1.5)
    assert jnp.allclose(region[1], 1.0 / 2.0)
    assert jnp.allclose(region[2], 1.0 / 3.0)


def test_apply_params_etched_reset_is_idempotent():
    # A background-dependent device must interpolate against the ORIGINAL background
    # each time; re-applying the same params must not compound.
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None)
    volume = SimulationVolume(partial_grid_shape=(8, 8, 8), material=Material(permittivity=2.0))
    device = _device(cls=EtchedDevice, materials={"etch": Material(permittivity=4.0)})
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects(
        object_list=[volume, device],
        config=config,
        constraints=[device.place_at_center(volume)],
        key=key,
    )
    assert arrays.initial_inv_permittivities is not None  # backup allocated for the etched device
    params["Device"] = jnp.full(jnp.asarray(params["Device"]).shape, 0.5, dtype=jnp.float32)
    arrays1, objects, _ = apply_params(arrays, objects, params, key)
    arrays2, objects, _ = apply_params(arrays1, objects, params, key)
    placed = objects["Device"]
    region1 = arrays1.inv_permittivities[:, *placed.grid_slice]
    region2 = arrays2.inv_permittivities[:, *placed.grid_slice]
    assert jnp.allclose(region1, region2)
    assert jnp.allclose(region2, 1.0 / 3.0)


def test_apply_params_device_dispersive_continuous_interpolates_coefficients():
    disp = DispersionModel(poles=(LorentzPole(resonance_frequency=2e14, damping=1e13, delta_epsilon=2.0),))
    materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=4.0, dispersion=disp)}
    placed, arrays, config = _apply_device_params(_device(materials=materials), Material(permittivity=1.0), 0.5)
    assert arrays.dispersive_c1 is not None
    num_poles = arrays.dispersive_c1.shape[0]
    num_comp = arrays.dispersive_c1.shape[1]
    c1, _, _, _ = compute_allowed_dispersive_coefficients(
        materials, dt=config.time_step_duration, max_num_poles=num_poles, num_components=num_comp
    )
    # materials sorted by permittivity: air(1.0) idx 0 (non-dispersive -> c1=0), si(4.0) idx 1 has the pole.
    # Continuous 50% mix -> region c1 == 0.5 * c1_si.
    c1_si = jnp.asarray(c1[1], dtype=arrays.dispersive_c1.dtype)
    region = arrays.dispersive_c1[:, :, *placed.grid_slice]
    expected = jnp.broadcast_to(0.5 * c1_si[:, :, None, None, None], region.shape)
    assert jnp.allclose(region, expected, atol=1e-6)


def _grad_through_apply_params(device, volume_material, params_value):
    """Return grad of sum(inv_permittivities) w.r.t. the device params, through apply_params."""
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None)
    volume = SimulationVolume(partial_grid_shape=(8, 8, 8), material=volume_material)
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects(
        object_list=[volume, device],
        config=config,
        constraints=[device.place_at_center(volume)],
        key=key,
    )
    p0 = jnp.full(jnp.asarray(params["Device"]).shape, params_value, dtype=jnp.float32)

    def loss(p):
        cur = dict(params)
        cur["Device"] = p
        new_arrays, _, _ = apply_params(arrays, objects, cur, key)
        return jnp.sum(new_arrays.inv_permittivities)

    return jax.grad(loss)(p0)


def test_apply_params_continuous_gradient_flows():
    materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
    grad = _grad_through_apply_params(_device(materials=materials), Material(permittivity=1.0), 0.5)
    assert jnp.all(jnp.isfinite(grad))
    # A denser material lowers inv-permittivity, so increasing the weight decreases the loss.
    assert jnp.all(grad < 0)


def test_apply_params_discrete_gradient_flows_via_ste():
    materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
    device = _device(materials=materials, param_transforms=[ClosestIndex()])
    grad = _grad_through_apply_params(device, Material(permittivity=1.0), 0.7)
    # The straight-through estimator keeps a finite, non-zero gradient despite the hard selection.
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0)


def test_etched_device_requires_continuous_output():
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None)
    device = _device(
        cls=EtchedDevice,
        materials={"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)},
        param_transforms=[ClosestIndex()],
    )
    with pytest.raises(Exception, match="only support continuous"):
        device.place_on_grid(grid_slice_tuple=((0, 4), (0, 4), (0, 4)), config=config, key=jax.random.PRNGKey(0))


def test_etched_device_requires_single_material():
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None)
    device = _device(
        cls=EtchedDevice,
        materials={"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)},
    )
    with pytest.raises(Exception, match="one material in etched device"):
        device.place_on_grid(grid_slice_tuple=((0, 4), (0, 4), (0, 4)), config=config, key=jax.random.PRNGKey(0))


def test_standard_device_continuous_requires_two_materials():
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None)
    device = _device(materials={"only": Material(permittivity=4.0)})
    with pytest.raises(Exception, match="exactly two materials"):
        device.place_on_grid(grid_slice_tuple=((0, 4), (0, 4), (0, 4)), config=config, key=jax.random.PRNGKey(0))


def test_apply_params_no_devices_leaves_arrays_unchanged():
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", gradient_config=None)
    volume = SimulationVolume(partial_grid_shape=(8, 8, 8), material=Material(permittivity=2.0))
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = place_objects(object_list=[volume], config=config, constraints=[], key=key)
    before = arrays.inv_permittivities
    arrays, objects, info = apply_params(arrays, objects, params, key)
    assert isinstance(info, dict)
    assert jnp.allclose(arrays.inv_permittivities, before)


# ---------------------------------------------------------------------------
# _apply_position_constraint - edge cases
# ---------------------------------------------------------------------------


def test_apply_position_constraint_other_not_placed(simple_config, simple_volume, simple_material):
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    """Position constraint skipped when other_object position unknown."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[None, None], [None, None], [None, None]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    c = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],
        grid_margins=[0],
        margins=[None],
    )
    resolved, _ = _apply_position_constraint(c, obj_map, config, shape_dict, slice_dict)
    assert resolved is False


def test_apply_position_constraint_object_size_unknown(simple_config, simple_volume, simple_material):
    """Position constraint skipped when object size unknown."""
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    c = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],
        grid_margins=[0],
        margins=[None],
    )
    resolved, _ = _apply_position_constraint(c, obj_map, config, shape_dict, slice_dict)
    assert resolved is False


# ---------------------------------------------------------------------------
# _apply_size_constraint - edge cases
# ---------------------------------------------------------------------------


def test_apply_size_constraint_other_shape_unknown(simple_config, simple_volume, simple_material):
    """Size constraint skipped when other object shape is unknown."""
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    c = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[0],
        offsets=[None],
    )
    resolved, _ = _apply_size_constraint(c, obj_map, config, shape_dict)
    assert resolved is False


def test_apply_size_constraint_conflicting_shape_raises(simple_config, simple_volume, simple_material):
    """Size constraint raises when computed shape conflicts with existing one."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Set obj1 shape to 60 in axis 0, but constraint would compute 50
    shape_dict = {"volume": [100, 100, 100], "obj1": [60, None, None]}
    c = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[0],
        offsets=[None],
    )
    with pytest.raises(Exception):
        _apply_size_constraint(c, obj_map, simple_config, shape_dict)


def test_apply_size_constraint_conflicting_shape_raises_descriptive(simple_config, simple_volume, simple_material):
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    """SizeConstraint that conflicts with an already-set shape raises an informative error."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [60, None, None]}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[0, 60], [None, None], [None, None]]}
    c = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],  # computes 50, but shape is already 60
        grid_offsets=[0],
        offsets=[None],
    )
    with pytest.raises(Exception, match="geometry"):
        _apply_size_constraint(c, obj_map, config, shape_dict, slice_dict)


# ---------------------------------------------------------------------------
# Cylinder __post_init__ — derived axis conflict tests
# ---------------------------------------------------------------------------


def test_cylinder_partial_grid_shape_on_derived_axis_raises(simple_material):
    """Cylinder rejects partial_grid_shape for a cross-section axis at construction time."""
    from fdtdx.objects.static_material.cylinder import Cylinder

    materials = {"mat": simple_material}
    with pytest.raises(Exception, match="derived from the radius"):
        Cylinder(
            name="cyl",
            radius=5.0,
            axis=2,
            material_name="mat",
            materials=materials,
            partial_grid_shape=(10, None, None),
        )


def test_cylinder_partial_real_shape_on_derived_axis_raises(simple_material):
    """Cylinder rejects partial_real_shape for a cross-section axis at construction time."""
    from fdtdx.objects.static_material.cylinder import Cylinder

    materials = {"mat": simple_material}
    with pytest.raises(Exception, match="derived from the radius"):
        Cylinder(
            name="cyl",
            radius=5.0,
            axis=2,
            material_name="mat",
            materials=materials,
            partial_real_shape=(8.0, None, None),
        )


def test_cylinder_sets_cross_section_from_radius(simple_material):
    """Cylinder.__post_init__ fills partial_real_shape for cross-section axes from radius."""
    from fdtdx.objects.static_material.cylinder import Cylinder

    materials = {"mat": simple_material}
    cyl = Cylinder(name="cyl", radius=5.0, axis=2, material_name="mat", materials=materials)
    assert cyl.partial_real_shape[0] == pytest.approx(10.0)
    assert cyl.partial_real_shape[1] == pytest.approx(10.0)
    assert cyl.partial_real_shape[2] is None  # extrusion axis still free


def test_cylinder_extrusion_axis_partial_real_shape_accepted(simple_material):
    """Cylinder accepts partial_real_shape for the extrusion axis."""
    from fdtdx.objects.static_material.cylinder import Cylinder

    materials = {"mat": simple_material}
    cyl = Cylinder(
        name="cyl",
        radius=5.0,
        axis=2,
        material_name="mat",
        materials=materials,
        partial_real_shape=(None, None, 20.0),
    )
    assert cyl.partial_real_shape[2] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# _update_grid_slices_from_shapes - direct unit tests
# ---------------------------------------------------------------------------


def test_update_grid_slices_from_shapes_b0_known(simple_volume, simple_material):
    """When b0 is known and shape known, b1 should be set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, None], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_slices, _new_errors = _update_grid_slices_from_shapes(obj_map, shape_dict, slice_dict, errors)
    assert resolved is True
    assert new_slices["obj1"][0][1] == 15  # b0(5) + shape(10)


def test_update_grid_slices_from_shapes_b1_known(simple_volume, simple_material):
    """When b1 is known and shape known, b0 should be set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, 30], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_slices, _new_errors = _update_grid_slices_from_shapes(obj_map, shape_dict, slice_dict, errors)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 20  # b1(30) - shape(10)


def test_update_grid_slices_from_shapes_inconsistent_shape(simple_volume, simple_material):
    """When both bounds known but shape mismatches, error is set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    # b1 - b0 = 20, but shape = 10 → inconsistency
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, 25], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    _resolved, _new_slices, new_errors = _update_grid_slices_from_shapes(obj_map, shape_dict, slice_dict, errors)
    assert new_errors["obj1"] is not None


# ---------------------------------------------------------------------------
# _update_grid_shapes_from_slices - direct unit tests
# ---------------------------------------------------------------------------


def test_update_grid_shapes_from_slices_infers_shape(simple_volume, simple_material):
    """When both bounds known and shape unknown, shape is inferred."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[10, 30], [0, 100], [0, 100]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_shapes, _new_errors = _update_grid_shapes_from_slices(obj_map, shape_dict, slice_dict, errors)
    assert resolved is True
    assert new_shapes["obj1"][0] == 20


def test_update_grid_shapes_from_slices_inconsistent(simple_volume, simple_material):
    """When bounds imply a different shape, error is set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # shape says 10 but bounds say 20
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, None, None]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, 25], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    _resolved, _new_shapes, new_errors = _update_grid_shapes_from_slices(obj_map, shape_dict, slice_dict, errors)
    assert new_errors["obj1"] is not None


# ---------------------------------------------------------------------------
# _apply_position_constraint - conflict paths
# ---------------------------------------------------------------------------


def test_apply_position_constraint_conflicting_b0_raises(simple_config, simple_volume, simple_material):
    """Position constraint raises when computed b0 conflicts with existing b0."""
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Place obj1 lower bound at 5 (b0=5), but constraint would compute b0=0
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, None], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    # Constraint: obj1 left edge at volume left edge (b0=0 for obj1)
    c = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[-1.0],
        other_object_positions=[-1.0],
        grid_margins=[0],
        margins=[None],
    )
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_position_constraint(c, obj_map, config, shape_dict, slice_dict)


def test_apply_position_constraint_conflicting_b1_raises(simple_config, simple_volume, simple_material):
    """Position constraint raises when computed b1 conflicts with existing b1."""
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Place obj1: b0=None but b1=50, and constraint would compute b0=0→b1=10
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, 50], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    # Constraint: obj1 left edge at volume left edge → b0=0, b1=10
    c = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[-1.0],
        other_object_positions=[-1.0],
        grid_margins=[0],
        margins=[None],
    )
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_position_constraint(c, obj_map, config, shape_dict, slice_dict)


# ---------------------------------------------------------------------------
# resolve_object_constraints - unknown constraint type via iterative solver
# ---------------------------------------------------------------------------


def test_apply_size_extension_constraint_conflicting_value_raises(simple_config, simple_volume, simple_material):
    """Size extension raises when computed anchor conflicts with existing bound."""
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Extend obj1 to volume lower boundary (0), but obj1's lower bound is already set to 20
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[20, None], [None, None], [None, None]],
    }
    c = SizeExtensionConstraint(
        object="obj1",
        other_object=None,
        axis=0,
        direction="-",
        other_position=0.0,
        grid_offset=None,
        offset=None,
    )
    with pytest.raises(Exception, match="Inconsistent grid shape"):
        _apply_size_extension_constraint(c, obj_map, config, slice_dict, "volume")


def test_apply_size_extension_constraint_volume_upper_bound_none_raises(simple_config, simple_volume, simple_material):
    """Raises when volume's upper bound is None (should never happen in normal flow)."""
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Volume upper bound is None (abnormal state)
    slice_dict = {
        "volume": [[0, None], [0, None], [0, None]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    c = SizeExtensionConstraint(
        object="obj1",
        other_object=None,
        axis=0,
        direction="+",
        other_position=0.0,
        grid_offset=None,
        offset=None,
    )
    with pytest.raises(Exception, match="This should never happen"):
        _apply_size_extension_constraint(c, obj_map, config, slice_dict, "volume")


# ---------------------------------------------------------------------------
# resolve_object_constraints - unknown constraint type via iterative solver
# ---------------------------------------------------------------------------


def test_resolve_constraints_unknown_constraint_type_sets_error(simple_config, simple_volume, simple_material):
    """An unrecognised constraint type is caught and stored as an error."""

    class FakeConstraint:
        object = "obj1"
        other_object = None

    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    fake = FakeConstraint()
    # _check_objects_names_from_constraints inspects getattr(c, "object") and getattr(c, "other_object")
    _resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [fake], simple_config)
    # The unknown type raises inside the loop, exception is caught → error stored
    assert errors["obj1"] is not None


# ---------------------------------------------------------------------------
# _init_arrays - unknown static material object type
# ---------------------------------------------------------------------------


@patch("fdtdx.fdtd.initialization.create_named_sharded_matrix")
def test_init_arrays_unknown_static_material_type_raises(mock_create_matrix):
    """_init_arrays should raise Exception for an unknown static material object type.

    Covers line 530 (else: raise Exception("Unknown object type")).
    The static_material_objects property normally only returns UniformMaterialObject
    or StaticMultiMaterialObject instances. This test calls _init_arrays directly
    with a mocked ObjectContainer to exercise the defensive else branch.
    """
    mock_create_matrix.side_effect = lambda shape, **kwargs: jnp.zeros(shape)

    class FakeStaticObj:
        placement_order = 0

    fake_obj = FakeStaticObj()

    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.uniform_spacing.return_value = 1.0
    config.backend = "cpu"
    config.gradient_config = None
    config.grid = UniformGrid(spacing=1.0)
    config.resolve_grid.return_value = RectilinearGrid.uniform(shape=(2, 2, 2), spacing=1.0)
    config.uniform_spacing.return_value = 1.0

    objects = Mock(spec=ObjectContainer)
    objects.volume = Mock()
    objects.volume.grid_shape = (2, 2, 2)
    objects.all_objects_isotropic_permittivity = True
    objects.all_objects_isotropic_permeability = True
    objects.all_objects_isotropic_electric_conductivity = True
    objects.all_objects_isotropic_magnetic_conductivity = True
    objects.all_objects_diagonally_anisotropic_permittivity = True
    objects.all_objects_diagonally_anisotropic_permeability = True
    objects.all_objects_diagonally_anisotropic_electric_conductivity = True
    objects.all_objects_diagonally_anisotropic_magnetic_conductivity = True
    objects.all_objects_non_magnetic = True
    objects.all_objects_non_electrically_conductive = True
    objects.all_objects_non_magnetically_conductive = True
    objects.max_num_dispersive_poles = 0
    objects.static_material_objects = [fake_obj]
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = []

    with pytest.raises(Exception, match="Unknown object type"):
        _init_arrays(objects, config)


# ---------------------------------------------------------------------------
# partial_real_position tests (from upstream)
# ---------------------------------------------------------------------------


def test_resolve_constraints_with_partial_real_position(simple_material):
    """Test resolving objects with partial_real_position specified.

    This test verifies that:
    - Real-world positions (centers) are correctly converted to grid coordinates
    - All three axes are properly positioned when specified
    - The conversion uses the simulation resolution correctly
    - Boundaries are computed from center position and size
    """
    config = SimulationConfig(grid=UniformGrid(spacing=0.5), time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(100, 100, 100),
    )

    # Object with real position (CENTER) specified
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(-15.0, -10.0, -5.0),  # Center positions in real-world coordinates
        partial_grid_shape=(10, 10, 10),  # Known size
        material=simple_material,
    )

    objects = [volume, obj]
    constraints = []  # No additional constraints needed

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed
    assert errors["obj1"] is None
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices

    # Verify conversion from real to grid coordinates
    # Real center 10.0 / resolution 0.5 = grid center 20
    # With size 10: lower = 20 - 5 = 15, upper = 15 + 10 = 25
    assert resolved_slices["obj1"][0] == (15, 25)

    # Real center 15.0 / resolution 0.5 = grid center 30
    # With size 10: lower = 30 - 5 = 25, upper = 25 + 10 = 35
    assert resolved_slices["obj1"][1] == (25, 35)

    # Real center 20.0 / resolution 0.5 = grid center 40
    # With size 10: lower = 40 - 5 = 35, upper = 35 + 10 = 45
    assert resolved_slices["obj1"][2] == (35, 45)


def test_resolve_constraints_partial_real_position_with_grid_shape(simple_config, simple_volume, simple_material):
    """Test partial_real_position works with partial_grid_shape.

    This test verifies that:
    - Axes with None in partial_real_position extend from 0 when size is known
    - Axes with specified center positions are correctly placed
    - Mixed specification (some axes positioned, others extending) works correctly
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(-30.0, None, -20.0),  # Only x and z centers specified
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: center at 20, size 10, half=5, so lower = 15, upper = 25
    assert resolved_slices["obj1"][0] == (15, 25)

    # y-axis: no position specified, extends from 0 with size 10
    assert resolved_slices["obj1"][1] == (0, 10)

    # z-axis: center at 30, size 10, half=5, so lower = 25, upper = 35
    assert resolved_slices["obj1"][2] == (25, 35)


def test_resolve_constraints_partial_real_position_conflicts_with_constraint(
    simple_config, simple_volume, simple_material
):
    """Test that conflicting partial_real_position and grid constraint raises error.

    This test verifies that:
    - Conflicts between partial_real_position and GridCoordinateConstraint are detected
    - The error is properly reported in the errors dictionary
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(15.0, None, None),  # Center at 15
        partial_grid_shape=(10, 10, 10),  # Size 10, so bounds should be (10, 20)
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Constraint that conflicts with partial_real_position
    # partial_real_position says lower bound should be 10, but constraint says 5
    constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[5],  # Conflicts with computed lower bound of 10
    )

    constraints = [constraint]

    _resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should detect the conflict
    assert errors["obj1"] is not None


def test_resolve_constraints_partial_real_position_with_real_shape(simple_material):
    """Test partial_real_position works together with partial_real_shape.

    This test verifies that:
    - Both partial_real_position (center) and partial_real_shape use consistent resolution
    - Objects can be fully specified using only real-world coordinates
    - The conversion to grid coordinates is accurate for both position and size
    """
    config = SimulationConfig(grid=UniformGrid(spacing=0.25), time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(200, 200, 200),
    )

    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(-15.0, -10.0, -5.0),  # Center positions
        partial_real_shape=(5.0, 10.0, 15.0),  # Sizes
        material=simple_material,
    )

    objects = [volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed
    assert errors["obj1"] is None

    # Verify conversions with resolution 0.25:
    # x: center 10.0/0.25=40, size 5.0/0.25=20, half=10 -> (30, 50)
    assert resolved_slices["obj1"][0] == (30, 50)

    # y: center 15.0/0.25=60, size 10.0/0.25=40, half=20 -> (40, 80)
    assert resolved_slices["obj1"][1] == (40, 80)

    # z: center 20.0/0.25=80, size 15.0/0.25=60, half=30 -> (50, 110)
    assert resolved_slices["obj1"][2] == (50, 110)


def test_resolve_constraints_partial_real_position_mixed_with_constraints(
    simple_config, simple_volume, simple_material
):
    """Test partial_real_position works alongside other constraints.

    This test verifies that:
    - Objects with partial_real_position can serve as anchors for positioning other objects
    - PositionConstraint properly references objects positioned via partial_real_position
    - The constraint resolution system integrates seamlessly with the new feature
    """
    obj1 = UniformMaterialObject(
        name="obj1",
        partial_real_position=(-35.0, None, None),  # Center at 15, size 10 -> bounds (10, 20)
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position obj2 relative to obj1
    constraint = PositionConstraint(
        object="obj2",
        other_object="obj1",
        axes=[0],
        object_positions=[-1.0],  # Left side of obj2
        other_object_positions=[1.0],  # Right side of obj1 (at 20)
        grid_margins=[5],
        margins=[None],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None
    assert errors["obj2"] is None

    # obj1 center at 15, size 10, half=5 -> (10, 20)
    assert resolved_slices["obj1"][0] == (10, 20)

    # obj1 y and z axes extend from 0 with size 10
    assert resolved_slices["obj1"][1] == (0, 10)
    assert resolved_slices["obj1"][2] == (0, 10)

    # obj2 should be positioned relative to obj1's right side (20) + margin (5)
    # obj2's left side at position 25, size 20, so ends at 45
    assert resolved_slices["obj2"][0] == (25, 45)


def test_resolve_constraints_partial_real_position_all_none(simple_config, simple_volume, simple_material):
    """Test object with partial_real_position=(None, None, None).

    This test verifies that:
    - Objects with all None values in partial_real_position work correctly
    - Such objects extend from 0 when size is known
    - No errors occur when partial_real_position is effectively unused
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(None, None, None),  # All axes unspecified
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # All axes extend from 0 with the specified size
    assert resolved_slices["obj1"][0] == (0, 10)
    assert resolved_slices["obj1"][1] == (0, 10)
    assert resolved_slices["obj1"][2] == (0, 10)


def test_resolve_constraints_partial_real_position_single_axis(simple_config, simple_volume, simple_material):
    """Test partial_real_position with only one axis specified.

    This test verifies that:
    - Single-axis center position specification works correctly
    - Other axes with None extend from 0 when size is known
    - The behavior matches expectations for partially-constrained objects
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(None, -30.0, None),  # Only y-axis center specified
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: extends from 0 with size 10
    assert resolved_slices["obj1"][0] == (0, 10)

    # y-axis: center at 20, size 10, half=5 -> (15, 25)
    assert resolved_slices["obj1"][1] == (15, 25)

    # z-axis: extends from 0 with size 10
    assert resolved_slices["obj1"][2] == (0, 10)


def test_resolve_constraints_partial_real_position_with_different_resolutions(simple_material):
    """Test partial_real_position with various resolution values.

    This test verifies that:
    - The resolution scaling works correctly for different values
    - Rounding to grid coordinates is consistent
    - Fine resolutions work properly with non-integer real positions
    """
    # Test with very fine resolution
    config_fine = SimulationConfig(grid=UniformGrid(spacing=0.1), time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(500, 500, 500),
    )

    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(-20.0, -15.0, -10.0),  # Center positions
        partial_grid_shape=(20, 20, 20),  # Size 20
        material=simple_material,
    )

    objects = [volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config_fine)

    # Should succeed
    assert errors["obj1"] is None

    # Verify: center 5.0/0.1=50, size 20, half=10 -> (40, 60)
    assert resolved_slices["obj1"][0] == (40, 60)

    # center 10.0/0.1=100, size 20, half=10 -> (90, 110)
    assert resolved_slices["obj1"][1] == (90, 110)

    # center 15.0/0.1=150, size 20, half=10 -> (140, 160)
    assert resolved_slices["obj1"][2] == (140, 160)


def test_resolve_constraints_partial_real_position_without_size(simple_config, simple_volume, simple_material):
    """Test partial_real_position without a known size.

    This test verifies that:
    - When center position is specified but size is not, we can't compute boundaries
    - The position information is ignored when size is unknown
    - Object extends to volume boundaries
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(50.0, 50.0, 50.0),  # Center specified
        # No size specified
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Without size, we can't compute boundaries from center
    # Object will extend to volume boundaries
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][0] == (0, 100)
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_resolve_constraints_partial_real_position_odd_size(simple_config, simple_volume, simple_material):
    """Test partial_real_position with odd-sized objects.

    This test verifies proper rounding when size/2 is not an integer.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(0.0, 0.0, 0.0),
        partial_grid_shape=(11, 13, 15),  # Odd sizes
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x: center 50, size 11, half=5.5 -> round(50-5.5)=44, upper=44+11=55
    assert resolved_slices["obj1"][0] == (44, 55)

    # y: center 50, size 13, half=6.5 -> round(50-6.5)=43, upper=43+13=56
    assert resolved_slices["obj1"][1] == (43, 56)

    # z: center 50, size 15, half=7.5 -> round(50-7.5)=42, upper=42+15=57
    assert resolved_slices["obj1"][2] == (42, 57)


# ---------------------------------------------------------------------------
# _extend_to_inf_if_possible boundary logic tests (from upstream)
# ---------------------------------------------------------------------------


def test_extend_to_inf_lower_bound_known_upper_not(simple_config, simple_volume, simple_material):
    """Test _extend_to_inf_if_possible when lower bound is known but upper is not.

    This covers the branch: elif b0 is not None and b1 is None and size is not None

    Scenario: Object has a known size and lower boundary set (via constraint),
    so upper boundary can be computed (should NOT extend upper boundary).
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, 15, 15),  # Known size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set lower boundaries only using GridCoordinateConstraint
    constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0, 1, 2],
        sides=["-", "-", "-"],  # Lower bounds
        coordinates=[10, 20, 30],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # Lower bounds from constraint
    assert resolved_slices["obj1"][0][0] == 10
    assert resolved_slices["obj1"][1][0] == 20
    assert resolved_slices["obj1"][2][0] == 30

    # Upper bounds computed as lower + size (NOT extended to volume boundary)
    assert resolved_slices["obj1"][0][1] == 25  # 10 + 15
    assert resolved_slices["obj1"][1][1] == 35  # 20 + 15
    assert resolved_slices["obj1"][2][1] == 45  # 30 + 15


def test_extend_to_inf_upper_bound_known_lower_not(simple_config, simple_volume, simple_material):
    """Test _extend_to_inf_if_possible when upper bound is known but lower is not.

    This covers the branch: elif b1 is not None and b0 is None and size is not None

    Scenario: Object has a known size and upper boundary set, so lower boundary
    can be computed (should NOT extend lower boundary).
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, 15, 15),  # Known size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set upper boundaries using GridCoordinateConstraint
    constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0, 1, 2],
        sides=["+", "+", "+"],  # Upper bounds
        coordinates=[50, 60, 70],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # Lower bounds computed as upper - size (NOT extended to 0)
    assert resolved_slices["obj1"][0][0] == 35  # 50 - 15
    assert resolved_slices["obj1"][1][0] == 45  # 60 - 15
    assert resolved_slices["obj1"][2][0] == 55  # 70 - 15

    # Upper bounds from constraint
    assert resolved_slices["obj1"][0][1] == 50
    assert resolved_slices["obj1"][1][1] == 60
    assert resolved_slices["obj1"][2][1] == 70


def test_extend_to_inf_partial_real_position_sets_both_bounds(simple_config, simple_volume, simple_material):
    """Test that partial_real_position with known size sets both boundaries.

    When partial_real_position (center) and size are both known,
    both boundaries are computed immediately, so this doesn't exercise
    the lower-bound-only or upper-bound-only branches.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(5.0, None, 5.0),  # Centers for x and z
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: center 55, size 20, half=10 -> (45, 65) - both bounds set
    assert resolved_slices["obj1"][0] == (45, 65)

    # y-axis: no center, extends from 0 with size 20
    assert resolved_slices["obj1"][1] == (0, 20)

    # z-axis: center 55, size 20, half=10 -> (45, 65) - both bounds set
    assert resolved_slices["obj1"][2] == (45, 65)


def test_extend_to_inf_with_real_coordinate_constraint_upper(simple_config, simple_volume, simple_material):
    """Test using RealCoordinateConstraint to set upper boundary.

    Another way to cover the upper-bound-known case.
    """
    config = _resolve_grid_from_volume([simple_volume], simple_config)
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set upper boundary using RealCoordinateConstraint
    constraint = RealCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["+"],  # Upper bound
        coordinates=[30.0],  # Real coordinate 30.0 / resolution 1.0: rectilinear grid 30.0 + 50.0 = 80
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower computed as 80 - 20 = 60
    assert resolved_slices["obj1"][0] == (60, 80)

    # y and z axes: extend from 0
    assert resolved_slices["obj1"][1] == (0, 20)
    assert resolved_slices["obj1"][2] == (0, 20)


def test_extend_to_inf_single_axis_upper_bound(simple_config, simple_volume, simple_material):
    """Test mixed scenario: some axes with lower bounds, one with upper bound.

    This exercises both branches in a single test.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, 15, 15),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set lower bounds for x and z, upper bound for y
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 2], sides=["-", "-"], coordinates=[10, 30]),
        GridCoordinateConstraint(object="obj1", axes=[1], sides=["+"], coordinates=[50]),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower bound known (10), upper computed (25) - covers first branch
    assert resolved_slices["obj1"][0] == (10, 25)

    # y-axis: upper bound known (50), lower computed (35) - covers second branch
    assert resolved_slices["obj1"][1] == (35, 50)

    # z-axis: lower bound known (30), upper computed (45) - covers first branch again
    assert resolved_slices["obj1"][2] == (30, 45)


def test_extend_to_inf_with_position_constraint_creating_lower_bound(simple_config, simple_volume, simple_material):
    """Test PositionConstraint creating a lower boundary situation.

    PositionConstraint can set both bounds at once when size is known,
    but if applied iteratively, might create lower-bound-only scenarios.
    """
    obj1 = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position obj1 first
    constraint1 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])

    # Position obj2 after obj1 - this sets obj2's lower bound
    constraint2 = PositionConstraint(
        object="obj2",
        other_object="obj1",
        axes=[0],
        object_positions=[-1.0],  # Left side of obj2
        other_object_positions=[1.0],  # Right side of obj1
        grid_margins=[5],
        margins=[None],
    )

    constraints = [constraint1, constraint2]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None
    assert errors["obj2"] is None

    # obj1: lower at 10, size 10, so upper at 20
    assert resolved_slices["obj1"][0] == (10, 20)

    # obj2: positioned after obj1 (20) + margin (5) = 25, size 20, so upper at 45
    assert resolved_slices["obj2"][0] == (25, 45)


def test_extend_to_inf_lower_bound_one_axis_only(simple_config, simple_volume, simple_material):
    """Test lower bound on just one axis to ensure the if check is covered.

    This specifically tests that (o, 1) is checked before removal.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, None, None),  # Only x has size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set lower boundary only for x-axis
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower=10, size=15, upper=25
    assert resolved_slices["obj1"][0] == (10, 25)

    # y and z extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_upper_bound_one_axis_only(simple_config, simple_volume, simple_material):
    """Test upper bound on just one axis to ensure the if check is covered.

    This specifically tests that (o, 0) is checked before removal.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, None, None),  # Only x has size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set upper boundary only for x-axis
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["+"], coordinates=[50])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: upper=50, size=15, lower=35
    assert resolved_slices["obj1"][0] == (35, 50)

    # y and z extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


# ---------------------------------------------------------------------------
# Complex iterative constraint resolution tests (from upstream)
# ---------------------------------------------------------------------------


def test_size_dependent_position_with_partial_real_position(simple_config, simple_volume, simple_material):
    """Test scenario where position depends on size that is resolved through constraints.

    Setup:
    - Volume has fully specified size
    - Object 2 has size constraint dependent on volume (half the size)
    - Object 2 position is specified through place_at_center (partial_real_position)
    - Object 3 has same_size constraint relative to object 2
    - Object 3 position is specified through partial_real_position

    This tests that the code can handle cases where:
    1. Size of object 2 is not known initially but resolved through SizeConstraint
    2. Position of object 2 must wait for size to be resolved
    3. Size of object 3 depends on object 2's size
    4. Position of object 3 must wait for its size to be resolved
    """
    # Object 2: size depends on volume, position at center
    obj2 = UniformMaterialObject(
        name="obj2",
        partial_real_position=(0, 0, 0),
        material=simple_material,
    )

    # Object 3: size depends on obj2, position specified
    obj3 = UniformMaterialObject(
        name="obj3",
        partial_real_position=(-25.0, None, None),
        material=simple_material,
    )

    objects = [simple_volume, obj2, obj3]

    constraints = [
        # obj2 should be half the size of volume in all dimensions
        SizeConstraint(
            object="obj2",
            other_object="volume",
            axes=[0, 1, 2],
            other_axes=[0, 1, 2],
            proportions=[0.5, 0.5, 0.5],
            grid_offsets=[0, 0, 0],
            offsets=[None, None, None],
        ),
        # obj3 should have same size as obj2
        SizeConstraint(
            object="obj3",
            other_object="obj2",
            axes=[0, 1, 2],
            other_axes=[0, 1, 2],
            proportions=[1.0, 1.0, 1.0],
            grid_offsets=[0, 0, 0],
            offsets=[None, None, None],
        ),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj2"] is None
    assert errors["obj3"] is None

    # obj2: size = 100 * 0.5 = 50, center at 50, so bounds are (25, 75)
    assert resolved_slices["obj2"][0] == (25, 75)
    assert resolved_slices["obj2"][1] == (25, 75)
    assert resolved_slices["obj2"][2] == (25, 75)

    # obj3: size = 50 (same as obj2), x-center at 25 so bounds are (0, 50)
    # y and z extend from 0 since no position specified
    assert resolved_slices["obj3"][0] == (0, 50)
    assert resolved_slices["obj3"][1] == (0, 50)
    assert resolved_slices["obj3"][2] == (0, 50)


def test_extend_to_inf_both_boundaries_known(simple_config, simple_volume, simple_material):
    """Test extend-to-infinity when both boundaries are already known.

    This tests lines 1102-1106 where both b0 and b1 are not None.
    The object should not be extended.
    """
    obj = UniformMaterialObject(
        name="obj1",
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set both boundaries explicitly
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10]),
        GridCoordinateConstraint(object="obj1", axes=[0], sides=["+"], coordinates=[30]),
        # Don't set y and z boundaries - they should extend
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: both boundaries explicitly set, not extended
    assert resolved_slices["obj1"][0] == (10, 30)

    # y and z axes: no boundaries set, should extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_lower_known_upper_computed(simple_config, simple_volume, simple_material):
    """Test extend-to-infinity when lower bound is known and upper is computed from size.

    This tests lines 1108-1110 where b0 is not None, b1 is None, and size is not None.
    Upper boundary should be computed, not extended.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(25, None, None),  # Only x has known size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set only the lower boundary for x-axis
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower=15, size=25, so upper=40 (computed, not extended to 100)
    assert resolved_slices["obj1"][0] == (15, 40)

    # y and z: no size or boundaries, extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_upper_known_lower_computed(simple_config, simple_volume, simple_material):
    """Test extend-to-infinity when upper bound is known and lower is computed from size.

    This tests lines 1112-1114 where b1 is not None, b0 is None, and size is not None.
    Lower boundary should be computed, not extended.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(30, None, None),  # Only x has known size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set only the upper boundary for x-axis
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["+"], coordinates=[80])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: upper=80, size=30, so lower=50 (computed, not extended to 0)
    assert resolved_slices["obj1"][0] == (50, 80)

    # y and z: no size or boundaries, extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_no_boundaries_with_size(simple_config, simple_volume, simple_material):
    """Test extend-to-infinity when no boundaries are set but size is known.

    This tests lines 1116-1120 where b0 is None, b1 is None, and size is not None.
    Lower should extend from 0, upper should be computed from size.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(40, 40, 40),  # Size is known
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # No position constraints at all
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # All axes: no boundaries set, size=40, so extends from 0 to 40
    assert resolved_slices["obj1"][0] == (0, 40)
    assert resolved_slices["obj1"][1] == (0, 40)
    assert resolved_slices["obj1"][2] == (0, 40)


def test_extend_to_inf_no_bounds_but_size_known_triggers_upper_removal(simple_config, simple_volume, simple_material):
    """
    Trigger branch:
    b0 is None, b1 is None, size is not None
    and (o, 1) is in extension_obj.
    """

    obj = UniformMaterialObject(
        name="obj1",
        partial_real_shape=(10.0, 10.0, 10.0),  # size known
        # no position -> no bounds
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []  # important: no constraints that remove extension flags

    _resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # This configuration is valid, we just care about hitting the branch
    assert errors["obj1"] is None


# ---------------------------------------------------------------------------
# Initial position resolution tests (from upstream)
# ---------------------------------------------------------------------------


def test_initial_position_skipped_when_size_unknown(simple_config, simple_volume, simple_material):
    """Lines 558-574: partial_real_position is set but no size is known yet.

    When the object's size cannot be determined during the initial (pre-iteration)
    pass, _resolve_static_positions_initial must skip computing bounds so that the
    iterative pass can pick it up later once a SizeConstraint resolves the size.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(0, 0, 0),  # center known
        # No partial_grid_shape / partial_real_shape -> size unknown at init time
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # A SizeConstraint will resolve the size during the iterative phase.
    # After that, _resolve_static_positions_iterative places the object.
    size_constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0, 1, 2],
        other_axes=[0, 1, 2],
        proportions=[0.2, 0.2, 0.2],
        grid_offsets=[0, 0, 0],
        offsets=[None, None, None],
    )

    constraints = [size_constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # size = 100 * 0.2 = 20; center = 50; half = 10 -> (40, 60)
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][0] == (40, 60)
    assert resolved_slices["obj1"][1] == (40, 60)
    assert resolved_slices["obj1"][2] == (40, 60)


def test_initial_position_skipped_when_axis_is_none(simple_config, simple_volume, simple_material):
    """Lines 558-574: partial_real_position has None for some axes.

    Axes whose partial_real_position entry is None must be left untouched by
    _resolve_static_positions_initial; only axes with an explicit value are placed.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(-20, None, 20),  # y intentionally absent
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    assert errors["obj1"] is None
    # x: center 30, size 10, half 5 -> (25, 35)
    assert resolved_slices["obj1"][0] == (25, 35)
    # y: no center specified -> extends from 0 with known size 10
    assert resolved_slices["obj1"][1] == (0, 10)
    # z: center 70, size 10, half 5 -> (65, 75)
    assert resolved_slices["obj1"][2] == (65, 75)


def test_initial_position_no_partial_real_position_attribute(simple_config, simple_volume, simple_material):
    """Lines 558-574: hasattr guard -- object without partial_real_position is silently skipped.

    _resolve_static_positions_initial uses hasattr() before accessing the attribute.
    An object that has no partial_real_position at all must be resolved by other means
    without any AttributeError.
    """
    # UniformMaterialObject with only a grid shape and no position hint.
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    # No position constraint either -- the object will extend from 0.
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    assert errors["obj1"] is None
    # Size known, no position -> placed at origin
    assert resolved_slices["obj1"][0] == (0, 10)
    assert resolved_slices["obj1"][1] == (0, 10)
    assert resolved_slices["obj1"][2] == (0, 10)


def test_iterative_position_upper_bound_conflict(simple_config, simple_volume, simple_material):
    """Line 632: elif b1 != upper -- upper bound already set conflicts with the
    upper bound implied by partial_real_position.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(45.0, None, None),  # no size at init -> initial pass skips
        # NO partial_grid_shape -> size unknown until SizeConstraint resolves it
        material=simple_material,
    )

    objects = [simple_volume, obj]

    constraints = [
        # Resolves size=20 for all axes in the constraint loop (step 4, iter 1).
        SizeConstraint(
            object="obj1",
            other_object="volume",
            axes=[0, 1, 2],
            other_axes=[0, 1, 2],
            proportions=[0.2, 0.2, 0.2],
            grid_offsets=[0, 0, 0],
            offsets=[None, None, None],
        ),
        # Sets b1=60 on axis 0 in the constraint loop (step 4, iter 1).
        # center=45, size=20 -> upper=55 != 60 -> conflict on line 632 in iter 2.
        GridCoordinateConstraint(
            object="obj1",
            axes=[0],
            sides=["+"],
            coordinates=[60],
        ),
    ]

    _resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Conflict between b1=60 (from constraint) and upper=55 (from partial_real_position)
    # must be recorded in the errors dict.
    assert errors["obj1"] is not None


# ---------------------------------------------------------------------------
# SizeExtensionConstraint interaction with extend-to-inf (from upstream)
# ---------------------------------------------------------------------------


def test_extend_to_inf_size_extension_suppresses_extension_on_correct_axis(
    simple_config, simple_volume, simple_material
):
    """Lines 1096-1102: a SizeExtensionConstraint removes (obj1, dir) from
    extension_obj only on the axis it targets.

    Setup: obj1 has known size on axis 0 only. A SizeExtensionConstraint extends
    obj1 toward the volume boundary on axis 0 direction "+". Both obj1 and the
    volume are fully resolved from the start, so nothing changes in any iteration
    and _extend_to_inf_if_possible is called every iteration. The constraint must
    remove (obj1, 1) from extension_obj for axis 0 only; axes 1 and 2 remain free
    to extend to the volume boundary.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, None, None),  # size known only on axis 0
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # The volume's upper boundary on axis 0 is known from the start (it is the
    # volume itself).  obj1 has no position constraints, so after the initial
    # static shape resolution nothing further resolves -> not changed -> extend
    # to inf is called with this SizeExtensionConstraint in the list.
    constraints = [
        SizeExtensionConstraint(
            object="obj1",
            other_object=None,  # extend to volume boundary
            axis=0,
            direction="+",
            other_position=0.0,
            grid_offset=None,
            offset=None,
        ),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    assert errors["obj1"] is None
    # axis 0: SizeExtensionConstraint pins upper=100 (volume boundary);
    # lower = 100 - 20 = 80.
    assert resolved_slices["obj1"][0] == (80, 100)
    # axes 1 and 2 are NOT suppressed by the axis-0 constraint -> extend to (0, 100).
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_size_constraint_does_not_suppress_extension(simple_config, simple_volume, simple_material):
    """Lines 1096-1102: a SizeConstraint (not SizeExtensionConstraint) must NOT
    remove anything from extension_obj.

    The isinstance(c, SizeExtensionConstraint) guard means only SizeExtensionConstraint
    entries affect extension_obj. With only a SizeConstraint present, after it resolves
    the size in iteration 1 nothing further resolves, so _extend_to_inf_if_possible is
    called in iteration 2 with no entries removed -- obj1 extends from 0 on all axes.
    """
    obj = UniformMaterialObject(
        name="obj1",
        material=simple_material,  # no partial shape -- size must come from constraint
    )

    objects = [simple_volume, obj]

    constraints = [
        SizeConstraint(
            object="obj1",
            other_object="volume",
            axes=[0, 1, 2],
            other_axes=[0, 1, 2],
            proportions=[0.5, 0.5, 0.5],
            grid_offsets=[0, 0, 0],
            offsets=[None, None, None],
        ),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    assert errors["obj1"] is None
    # size=50, no position suppression -> extends from 0 on all axes.
    assert resolved_slices["obj1"][0] == (0, 50)
    assert resolved_slices["obj1"][1] == (0, 50)
    assert resolved_slices["obj1"][2] == (0, 50)


def test_extend_to_inf_size_extension_on_different_axis_does_not_suppress(
    simple_config, simple_volume, simple_material
):
    """Lines 1096-1102: axis == c.axis guard -- a SizeExtensionConstraint on axis 0
    must not suppress extension on axes 1 or 2.

    With only a SizeExtensionConstraint on axis 0, the loop at lines 1098-1102 removes
    (obj1, 1) from extension_obj only while processing axis 0. For axes 1 and 2 the
    condition `axis == c.axis` is False, so the removal is skipped and those axes
    extend freely to the volume boundary.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, None, None),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    constraints = [
        SizeExtensionConstraint(
            object="obj1",
            other_object=None,
            axis=0,
            direction="-",  # lower boundary on axis 0
            other_position=0.0,
            grid_offset=None,
            offset=None,
        ),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    assert errors["obj1"] is None
    # axis 0: lower bound pinned to volume boundary 0; upper = 0 + 20 = 20.
    assert resolved_slices["obj1"][0] == (0, 20)
    # axes 1 and 2: axis != 0, removal skipped -> extend to (0, 100).
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_lower_bound_only_already_removed_from_extension_obj(
    simple_config, simple_volume, simple_material
):
    """Lines 1107-1118: b0 not None, b1 None, size not None, but (o, 1) already
    absent from extension_obj because the SizeExtensionConstraint loop removed it.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, None, None),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    constraints = [
        # Sets b0=10 on axis 0; upper left None.
        GridCoordinateConstraint(
            object="obj1",
            axes=[0],
            sides=["-"],
            coordinates=[10],
        ),
        SizeExtensionConstraint(
            object="obj1",
            other_object=None,
            axis=0,
            direction="+",
            other_position=0.0,
            grid_offset=None,
            offset=None,
        ),
    ]

    _resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    assert "obj1" in errors


def test_warn_if_simulation_volume_too_large_emits_warning():
    """Volumes above MAX_SIMULATION_VOLUME_CELLS should warn before allocation."""
    grid_shape = (2200, 2200, 2200)  # 2154**3 > 10e9
    assert 2200**3 > MAX_SIMULATION_VOLUME_CELLS

    with pytest.warns(UserWarning, match="exceeds the recommended limit"):
        _warn_if_simulation_volume_too_large(grid_shape)


def test_warn_if_simulation_volume_at_limit_no_warning():
    """Volumes at or below the limit should not warn."""
    grid_shape = (2153, 2153, 2153)  # 2153**3 < 10e9
    assert 2153**3 <= MAX_SIMULATION_VOLUME_CELLS

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_simulation_volume_too_large(grid_shape)
