import math
import warnings
from abc import ABC
from typing import TYPE_CHECKING, NamedTuple, Self, Sequence, cast

import jax
import jax.numpy as jnp

from fdtdx import constants
from fdtdx.colors import XKCD_LIGHT_PINK, Color
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.jax.utils import check_specs
from fdtdx.core.misc import expand_matrix, invert_property, is_float_divisible
from fdtdx.materials import (
    Material,
    compute_allowed_dispersive_coefficients,
    compute_allowed_electric_conductivities,
    compute_allowed_permittivities,
)
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.object import OrderableObject
from fdtdx.typing import (
    INVALID_SHAPE_3D,
    UNDEFINED_SHAPE_3D,
    ParameterType,
    PartialGridShape3D,
    PartialRealShape3D,
    SliceTuple3D,
)

if TYPE_CHECKING:
    from fdtdx.fdtd.container import ArrayContainer


class _MaterialArraySlices(NamedTuple):
    """The material-property arrays a device writes into its own grid slice.

    Every field is shaped for the device's placed ``grid_slice``. ``inv_permittivity``
    has shape ``(num_components, *device_grid_shape)``; the dispersive coefficient
    slices (when not ``None``) have shape ``(num_poles, num_disp_components, *device_grid_shape)``.
    ``electric_conductivity`` (when not ``None``) has shape
    ``(num_cond_components, *device_grid_shape)`` and must already be scaled into the
    dimensionless update coefficient — see :meth:`Device.conductivity_scale`.
    """

    inv_permittivity: jax.Array
    dispersive_c1: jax.Array | None = None
    dispersive_c2: jax.Array | None = None
    dispersive_c3: jax.Array | None = None
    dispersive_c4: jax.Array | None = None
    electric_conductivity: jax.Array | None = None


@autoinit
class Device(OrderableObject, ABC):
    """Base class for devices with optimizable permittivity distributions.

    A device owns a latent parameter array, maps it through its
    ``param_transforms`` pipeline into per-voxel material weights/indices, and
    then materializes those into the simulation's inverse-permittivity (and,
    when present, dispersive ADE coefficient) arrays via :meth:`apply_to_arrays`.

    This class implements the common "select/interpolate between the device's own
    materials" behavior for both continuous and discrete parameterizations.
    Subclasses override :meth:`_compute_material_slices` (and, if needed,
    :meth:`_validate_materials` / :attr:`background_dependent`) to change how the
    material properties are written — see :class:`EtchedDevice` for an example
    that interpolates against the existing background material instead.
    """

    #: Dictionary of materials to be used in the device.
    materials: dict[str, Material] = field()

    #: A Sequence of parameter transformation to be applied to the parameters when mapping them to simulation materials.
    param_transforms: Sequence[ParameterTransformation] = field()

    #: Color of the object when plotted. Defaults to XKCD_LIGHT_PINK.
    color: Color | None = frozen_field(default=XKCD_LIGHT_PINK)

    #: Size of the material voxels used within the device in metrical units (meter). Note that this is independent of the simulation voxel size.
    #: Defaults to undefined shape. For all three axes, either the voxel grid or real shape needs to be defined.
    partial_voxel_grid_shape: PartialGridShape3D = frozen_field(default=UNDEFINED_SHAPE_3D)

    #: Size of the material voxels used within the device in simulation voxels. Defaults to undefined shape.
    #: For all three axes, either the voxel grid or real shape needs to be defined.
    partial_voxel_real_shape: PartialRealShape3D = frozen_field(default=UNDEFINED_SHAPE_3D)

    _single_voxel_grid_shape: tuple[int, int, int] = frozen_private_field(default=INVALID_SHAPE_3D)
    _matrix_voxel_grid_shape_override: tuple[int, int, int] = frozen_private_field(default=INVALID_SHAPE_3D)
    _physical_design_voxel_shape: tuple[float, float, float] | None = frozen_private_field(default=None)
    _physical_design_domain_shape: tuple[float, float, float] | None = frozen_private_field(default=None)

    # TODO(teevee112): support physical-unit design voxels on non-uniform grids — requires a resampling layer
    # or snapping to RectilinearGrid cell boundaries; currently only grid-cell-count voxels are
    # reliable on non-uniform grids (see PR #312)
    @property
    def matrix_voxel_grid_shape(self) -> tuple[int, int, int]:
        """Calculate the shape of the voxel matrix in grid coordinates.

        Returns:
            tuple[int, int, int]: Tuple of (x,y,z) dimensions representing how many voxels fit in each direction
                of the grid shape when divided by the single voxel shape.
        """
        return (
            self._matrix_voxel_grid_shape_override
            if self._matrix_voxel_grid_shape_override != INVALID_SHAPE_3D
            else (
                round(self.grid_shape[0] / self.single_voxel_grid_shape[0]),
                round(self.grid_shape[1] / self.single_voxel_grid_shape[1]),
                round(self.grid_shape[2] / self.single_voxel_grid_shape[2]),
            )
        )

    @property
    def single_voxel_grid_shape(self) -> tuple[int, int, int]:
        """Get the shape of a single voxel in grid coordinates.

        Returns:
            tuple[int, int, int]: Tuple of (x,y,z) dimensions for one voxel.
        """
        if self._single_voxel_grid_shape == INVALID_SHAPE_3D:
            raise Exception(f"{self} is not initialized yet")
        return self._single_voxel_grid_shape

    @property
    def single_voxel_real_shape(self) -> tuple[float, float, float]:
        """Calculate the representative physical size of one design voxel.

        Returns:
            Tuple of ``(x, y, z)`` dimensions in metres.

        Notes:
            On uniform simulation grids this is the exact size of each design
            voxel.  On non-uniform grids, devices are currently supported only
            when design voxels are specified by simulation-cell counts.  The
            returned physical size is then the average design-voxel extent over
            the placed device, suitable for transforms that need a representative
            scale.  True physical-size design voxels still require a resampling
            layer and are rejected during placement.
        """
        if self._physical_design_voxel_shape is not None:
            return self._physical_design_voxel_shape
        grid = self._config.resolved_grid
        if grid is not None and not grid.is_uniform:
            return (
                self.real_shape[0] / self.matrix_voxel_grid_shape[0],
                self.real_shape[1] / self.matrix_voxel_grid_shape[1],
                self.real_shape[2] / self.matrix_voxel_grid_shape[2],
            )

        single_voxel_shape = self.single_voxel_grid_shape
        spacing = self._config.uniform_spacing()
        return (
            single_voxel_shape[0] * spacing,
            single_voxel_shape[1] * spacing,
            single_voxel_shape[2] * spacing,
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        """Names of the arrays the parameter transformation pipeline must produce.

        The pipeline is seeded with one array of shape :attr:`matrix_voxel_grid_shape`
        per name, and :meth:`__call__` returns exactly these arrays. The default device
        drives every material property from a single weight array, so it needs only one
        channel. Subclasses that steer several properties independently override this to
        request one channel per property — see :class:`PermittivityConductivityDevice`.
        """
        return ("params",)

    @property
    def output_type(self) -> ParameterType:
        """The parameter type produced by the transformation pipeline.

        All channels in :attr:`output_names` must agree on one type: the material logic
        branches on the parameterization as a whole (interpolate vs. select), not per
        channel.
        """
        if not self.param_transforms:
            return ParameterType.CONTINUOUS
        out_type = self.param_transforms[-1]._output_type
        if not isinstance(out_type, dict):
            return out_type
        distinct_types = set(out_type.values())
        if len(distinct_types) != 1:
            raise Exception(
                f"The parameter mapping of {self.name} must produce arrays that all share a single "
                f"ParameterType, but got: {out_type}"
            )
        return next(iter(distinct_types))

    @property
    def conductivity_scale(self) -> float:
        """Factor converting a physical conductivity into the FDTD update coefficient.

        Mirrors the ``conductivity_spacing`` scaling that
        :func:`fdtdx.fdtd.initialization._init_arrays` applies to static objects, so a
        device writing ``electric_conductivity`` lands on the same scale. On uniform
        grids this equals the grid spacing; on stretched grids it is the reference
        spacing implied by ``c0 * dt / courant``.
        """
        return constants.c * self._config.time_step_duration / self._config.courant_number

    @property
    def background_dependent(self) -> bool:
        """Whether this device reads the existing permittivity inside its region.

        When ``True``, :func:`fdtdx.fdtd.initialization.apply_params` keeps a
        pristine backup of the initial inverse permittivity and restores it before
        every application so the device always interpolates against the original
        background rather than its own previous output. The default standard device
        overwrites its region outright and therefore returns ``False``.
        """
        return False

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        # determine voxel shape
        voxel_grid_shape = []
        spacing = None
        if not config.has_nonuniform_grid:
            spacing = config.uniform_spacing()
        uses_physical_design_grid = config.has_nonuniform_grid and any(
            shape is not None for shape in self.partial_voxel_real_shape
        )
        if uses_physical_design_grid and any(shape is not None for shape in self.partial_voxel_grid_shape):
            raise ValueError(
                "Non-uniform physical device voxel sizes cannot be mixed with partial_voxel_grid_shape. "
                "Use either a physical design grid or grid-cell-count design voxels."
            )
        if uses_physical_design_grid:
            raise NotImplementedError(
                "Physical-unit design voxels (partial_voxel_real_shape) are not yet supported on "
                "non-uniform grids. Use partial_voxel_grid_shape to specify voxel sizes in "
                "simulation grid-cell counts instead."
            )
        physical_design_shape = []
        matrix_shape_override = []
        for axis in range(3):
            partial_grid = self.partial_voxel_grid_shape[axis]
            partial_real = self.partial_voxel_real_shape[axis]
            if partial_grid is not None and partial_real is not None:
                raise Exception(f"Multi-Material voxels overspecified in axis: {axis=}")
            if partial_grid is not None:
                voxel_grid_shape.append(partial_grid)
            elif partial_real is not None:
                if uses_physical_design_grid:
                    if partial_real <= 0:
                        raise ValueError(f"Physical design voxel size must be positive for {axis=}.")
                    physical_design_shape.append(float(partial_real))
                    matrix_shape_override.append(max(1, math.ceil(self.real_shape[axis] / partial_real)))
                    voxel_grid_shape.append(1)
                else:
                    assert spacing is not None
                    cell_count = round(partial_real / spacing)
                    if cell_count < 1:
                        raise ValueError(
                            f"Device voxel size {partial_real:.3e} m on axis {axis} rounds to 0 cells "
                            f"at spacing {spacing:.3e} m. Increase the voxel size or reduce the grid spacing."
                        )
                    voxel_grid_shape.append(cell_count)
            else:
                raise Exception(f"Multi-Material voxels not specified in axis: {axis=}")

        self = self.aset("_single_voxel_grid_shape", tuple(voxel_grid_shape))
        if uses_physical_design_grid:
            self = self.aset("_physical_design_voxel_shape", tuple(physical_design_shape))
            self = self.aset("_physical_design_domain_shape", self.real_shape)
            self = self.aset("_matrix_voxel_grid_shape_override", tuple(matrix_shape_override))

        # sanity checks on the voxel shape
        for axis in range(3):
            if spacing is not None:
                float_div = is_float_divisible(
                    self.single_voxel_real_shape[axis],
                    spacing,
                    tolerance=max(1e-15, abs(spacing) * 1e-4),
                )
                if not float_div:
                    raise Exception(f"Not divisible: {self.single_voxel_real_shape[axis]=}, {spacing=}")
            if not uses_physical_design_grid and self.grid_shape[axis] % self.matrix_voxel_grid_shape[axis] != 0:
                raise Exception(
                    f"Due to discretization, matrix got skewered for {axis=}. "
                    f"{self.grid_shape=}, {self.matrix_voxel_grid_shape=}"
                )

        # init parameter transformations
        # We need to go once backwards through the transformations to determine the shape of the latent parameters
        # then we need to go forward through the transformations again to determine the parameter type of the
        # output
        new_t_list: list[ParameterTransformation] = []
        cur_shape = {name: self.matrix_voxel_grid_shape for name in self.output_names}
        for transform in self.param_transforms[::-1]:
            t_new = transform.init_module(
                config=config,
                materials=self.materials,
                matrix_voxel_grid_shape=self.matrix_voxel_grid_shape,
                single_voxel_size=self.single_voxel_real_shape,
                output_shape=cast(dict[str, tuple[int, ...]], cur_shape),
            )
            new_t_list.append(t_new)
            cur_shape = t_new._input_shape

        # init shape of transformations by going backwards through new list.
        # cur_shape now describes the latent parameters entering the first transform, so
        # its keys are exactly the channels init_params will create.
        module_list: list[ParameterTransformation] = []
        cur_input_type = {name: ParameterType.CONTINUOUS for name in cur_shape}
        for transform in new_t_list[::-1]:
            t_new = transform.init_type(
                input_type=cur_input_type,
            )
            module_list.append(t_new)
            cur_input_type = t_new._output_type

        # set own input shape dtype
        self = self.aset("param_transforms", module_list)
        self._validate_materials()
        return self

    def _validate_materials(self) -> None:
        """Validate the material set against this device's output parameterization.

        The standard device interpolates between exactly two materials for
        continuous output and selects among any number of materials for discrete
        output. Subclasses override this to encode their own requirements.

        Raises:
            Exception: If the material count is incompatible with the output type.
        """
        if self.output_type == ParameterType.CONTINUOUS and len(self.materials) != 2:
            raise Exception(
                "Need exactly two materials in device when parameter mapping outputs continuous permittivity "
                f"indices, but got {self.materials}"
            )

    @staticmethod
    def _overlap_weights_1d(sim_edges: jax.Array, design_edges: jax.Array) -> jax.Array:
        """Return design-voxel overlap fractions for each simulation cell.

        Rows correspond to simulation cells and columns correspond to design
        voxels.  Each row sums to one for cells contained inside the local
        design domain.  Using overlap fractions instead of center sampling keeps
        physical-size design grids conservative on stretched meshes: a large
        simulation cell that straddles multiple design voxels receives the
        volume-weighted average of those parameters.
        """
        sim_lower = sim_edges[:-1, None]
        sim_upper = sim_edges[1:, None]
        design_lower = design_edges[None, :-1]
        design_upper = design_edges[None, 1:]
        overlap = jnp.maximum(0.0, jnp.minimum(sim_upper, design_upper) - jnp.maximum(sim_lower, design_lower))
        widths = sim_upper - sim_lower
        return overlap / widths

    def _resample_design_params_to_sim_grid(self, params: jax.Array) -> jax.Array:
        """Map design-grid parameters to simulation cells.

        Grid-count design voxels use the legacy repeat expansion.  Physical
        design voxels on non-uniform grids use separable volume-overlap weights
        so the expanded simulation grid represents the average design parameter
        over each rectilinear simulation cell.
        """
        grid = self._config.resolved_grid
        if self._physical_design_voxel_shape is None or grid is None:
            return expand_matrix(
                matrix=params,
                grid_points_per_voxel=self.single_voxel_grid_shape,
            )
        if self._physical_design_domain_shape is None:
            raise RuntimeError("Physical design-grid devices must be placed before expansion.")

        overlap_weights = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            sim_edges = grid.edges(axis)[lower : upper + 1]
            design_edges = jnp.linspace(
                0.0,
                self._physical_design_domain_shape[axis],
                self.matrix_voxel_grid_shape[axis] + 1,
                dtype=sim_edges.dtype,
            )
            local_sim_edges = sim_edges - sim_edges[0]
            overlap_weights.append(self._overlap_weights_1d(local_sim_edges, design_edges))
        return jnp.einsum("ia,jb,kc,abc->ijk", overlap_weights[0], overlap_weights[1], overlap_weights[2], params)

    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array] | jax.Array:
        if len(self.param_transforms) > 0:
            shapes = self.param_transforms[0]._input_shape
        else:
            shapes = {name: self.matrix_voxel_grid_shape for name in self.output_names}
        if not isinstance(shapes, dict):
            shapes = {"params": shapes}
        params = {}
        for k, cur_shape in shapes.items():
            key, subkey = jax.random.split(key)
            p = jax.random.uniform(
                key=subkey,
                shape=cur_shape,
                minval=0,  # parameter always live between 0 and 1
                maxval=1,
                dtype=jnp.float32,
            )
            params[k] = p
        if len(params) == 1:
            params = next(iter(params.values()))
        return params

    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        expand_to_sim_grid: bool = False,
        **transform_kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        """Map latent parameters through this device's transformation pipeline.

        Args:
            params (dict[str, jax.Array] | jax.Array): The device's latent parameters —
                a bare array for single-channel devices, or one array per latent channel.
            expand_to_sim_grid (bool): If True, resample every resulting channel from the
                device's design-voxel grid onto the simulation grid.
            **transform_kwargs: Extra keyword arguments forwarded to each transformation.

        Returns:
            dict[str, jax.Array] | jax.Array: The pipeline output — a bare array when the
            device has a single channel (the common case), otherwise one array per name in
            :attr:`output_names`.

        Raises:
            Exception: If the pipeline does not end with exactly :attr:`output_names`.
        """
        if not isinstance(params, dict):
            params = {"params": params}
        # walk through modules
        for transform in self.param_transforms:
            check_specs(params, transform._input_shape)
            params_dict = cast(dict[str, jax.Array], params)
            params = transform(params_dict, **transform_kwargs)
            check_specs(params, transform._output_shape)
        params = cast(dict[str, jax.Array], params)
        if set(params.keys()) != set(self.output_names):
            raise Exception(
                f"The parameter mapping of {self.name} should return exactly the arrays {self.output_names}, "
                f"but got {tuple(params.keys())}. If using a continuous device, please make sure that the "
                "latent transformations abide to this rule."
            )
        if expand_to_sim_grid:
            params = {k: self._resample_design_params_to_sim_grid(v) for k, v in params.items()}
        if len(self.output_names) == 1:
            return params[self.output_names[0]]
        return params

    def apply_to_arrays(
        self,
        arrays: "ArrayContainer",
        params: dict[str, jax.Array] | jax.Array,
        **transform_kwargs,
    ) -> "ArrayContainer":
        """Materialize this device's parameters into the simulation material arrays.

        Maps ``params`` through the parameter transformation pipeline and writes
        the resulting inverse permittivity — plus, when the device produces them, the
        electric conductivity and the per-cell ADE recurrence coefficients — into the
        device's own grid slice of ``arrays``. This is the device analogue of
        :meth:`~fdtdx.objects.object.SimulationObject.apply`: the device has full
        control over what it writes and is not limited to interpolation weights in
        ``[0, 1]``, real permittivities, or isotropic materials.

        Subclasses customize the written values by overriding
        :meth:`_compute_material_slices`; the mechanics of setting the slices into
        the container (including recomputing the cached ``1/c2``) live here so they
        stay consistent across device types.

        Args:
            arrays (ArrayContainer): Container holding the current material arrays.
            params (dict[str, jax.Array] | jax.Array): This device's latent parameters.
            **transform_kwargs: Extra keyword arguments forwarded to the parameter
                transformation pipeline (e.g. a projection ``beta``).

        Returns:
            ArrayContainer: A copy of ``arrays`` with this device's region updated.
        """
        slices = self._compute_material_slices(arrays, params, **transform_kwargs)

        new_inv_perm = arrays.inv_permittivities.at[:, *self.grid_slice].set(slices.inv_permittivity)
        arrays = arrays.at["inv_permittivities"].set(new_inv_perm)

        if slices.electric_conductivity is not None:
            # _compute_material_slices only produces this slice when the array exists;
            # allocation is decided sim-wide by _init_arrays from every object's materials.
            assert arrays.electric_conductivity is not None
            new_sigma_e = arrays.electric_conductivity.at[:, *self.grid_slice].set(slices.electric_conductivity)
            arrays = arrays.at["electric_conductivity"].set(new_sigma_e)

        if slices.dispersive_c1 is not None:
            assert (
                arrays.dispersive_c1 is not None
                and arrays.dispersive_c2 is not None
                and arrays.dispersive_c3 is not None
                and slices.dispersive_c2 is not None
                and slices.dispersive_c3 is not None
            )
            new_c1 = arrays.dispersive_c1.at[:, :, *self.grid_slice].set(slices.dispersive_c1)
            new_c2 = arrays.dispersive_c2.at[:, :, *self.grid_slice].set(slices.dispersive_c2)
            new_c3 = arrays.dispersive_c3.at[:, :, *self.grid_slice].set(slices.dispersive_c3)
            # Recompute inv_c2 from the post-update c2. Do NOT interpolate inv_c2
            # directly: 1/avg(c2) != avg(1/c2), and the reverse-time ADE relies on
            # inv_c2 being the exact reciprocal of the stored c2.
            new_inv_c2 = jnp.where(new_c2 == 0, 0.0, 1.0 / new_c2)
            arrays = arrays.at["dispersive_c1"].set(new_c1)
            arrays = arrays.at["dispersive_c2"].set(new_c2)
            arrays = arrays.at["dispersive_c3"].set(new_c3)
            arrays = arrays.at["dispersive_inv_c2"].set(new_inv_c2)
            if slices.dispersive_c4 is not None:
                assert arrays.dispersive_c4 is not None
                new_c4 = arrays.dispersive_c4.at[:, :, *self.grid_slice].set(slices.dispersive_c4)
                arrays = arrays.at["dispersive_c4"].set(new_c4)

        return arrays

    def _compute_material_slices(
        self,
        arrays: "ArrayContainer",
        params: dict[str, jax.Array] | jax.Array,
        **transform_kwargs,
    ) -> _MaterialArraySlices:
        """Compute the material-property slices this device writes into its region.

        The standard implementation interpolates (continuous output) between the
        device's two materials, or selects (discrete output, via a straight-through
        estimator) among any number of materials. The number of permittivity
        components (1 / 3 / 9) and dispersive poles is inferred from the shapes of
        the arrays already allocated for the whole simulation.

        Args:
            arrays (ArrayContainer): Container holding the current material arrays.
            params (dict[str, jax.Array] | jax.Array): This device's latent parameters.
            **transform_kwargs: Extra keyword arguments forwarded to ``__call__``.

        Returns:
            _MaterialArraySlices: The inverse permittivity and (optional) dispersive
            coefficient slices sized for this device's grid slice.
        """
        # Single-channel device: __call__ returns a bare array.
        cur_material_indices = cast(jax.Array, self(params, expand_to_sim_grid=True, **transform_kwargs))

        num_components = arrays.inv_permittivities.shape[0]
        isotropic = num_components == 1
        diagonally_anisotropic = num_components == 3

        # (num_materials, num_components)
        allowed_perm = jnp.asarray(
            compute_allowed_permittivities(
                self.materials,
                isotropic=isotropic,
                diagonally_anisotropic=diagonally_anisotropic,
            )
        )

        # Dispersive coefficients are written whenever the simulation allocated
        # them, even if this device's own materials are non-dispersive: the
        # zero-padded coefficients then overwrite any stale values inherited from
        # a dispersive region underneath the device.
        write_dispersive = arrays.dispersive_c1 is not None
        write_c4 = write_dispersive and arrays.dispersive_c4 is not None
        allowed_c1 = allowed_c2 = allowed_c3 = allowed_c4 = None
        if write_dispersive:
            assert arrays.dispersive_c1 is not None
            num_poles = arrays.dispersive_c1.shape[0]
            num_disp_components = arrays.dispersive_c1.shape[1]
            c1_np, c2_np, c3_np, c4_np = compute_allowed_dispersive_coefficients(
                self.materials,
                dt=self._config.time_step_duration,
                max_num_poles=num_poles,
                num_components=num_disp_components,
            )
            dtype = arrays.dispersive_c1.dtype
            allowed_c1 = jnp.asarray(c1_np, dtype=dtype)
            allowed_c2 = jnp.asarray(c2_np, dtype=dtype)
            allowed_c3 = jnp.asarray(c3_np, dtype=dtype)
            allowed_c4 = jnp.asarray(c4_np, dtype=dtype)

        c1_slice = c2_slice = c3_slice = c4_slice = None

        if self.output_type == ParameterType.CONTINUOUS:
            # Linear interpolation between the two device materials.
            perm_bc = allowed_perm[:, :, None, None, None]  # (2, num_components, 1, 1, 1)
            perm_slice = perm_bc[0] + cur_material_indices * (perm_bc[1] - perm_bc[0])
            inv_perm_slice = invert_property(perm_slice)
            if write_dispersive:
                assert allowed_c1 is not None and allowed_c2 is not None and allowed_c3 is not None
                # allowed_cN[i]: (num_poles, num_disp_components); broadcast over the grid.
                w0 = (1 - cur_material_indices)[None, None, ...]  # (1, 1, Nx, Ny, Nz)
                w1 = cur_material_indices[None, None, ...]
                c1_slice = w0 * allowed_c1[0][:, :, None, None, None] + w1 * allowed_c1[1][:, :, None, None, None]
                c2_slice = w0 * allowed_c2[0][:, :, None, None, None] + w1 * allowed_c2[1][:, :, None, None, None]
                c3_slice = w0 * allowed_c3[0][:, :, None, None, None] + w1 * allowed_c3[1][:, :, None, None, None]
                if write_c4:
                    assert allowed_c4 is not None
                    c4_slice = w0 * allowed_c4[0][:, :, None, None, None] + w1 * allowed_c4[1][:, :, None, None, None]
        else:
            # Discrete material selection. Precompute inverse permittivities since
            # the selection is a gather, then keep continuous gradients via STE.
            if isotropic or diagonally_anisotropic:
                inv_allowed = 1.0 / allowed_perm  # (num_materials, num_components)
            else:
                # Fully anisotropic: invert each 3x3 tensor and flatten back to 9 elements.
                inv_allowed = jnp.array([jnp.linalg.inv(p.reshape(3, 3)).flatten() for p in allowed_perm])
            int_idx = cur_material_indices.astype(jnp.int32)
            # inv_allowed[int_idx]: (*grid, num_components) -> (num_components, *grid)
            component_values = jnp.moveaxis(inv_allowed[int_idx], -1, 0)
            inv_perm_slice = straight_through_estimator(cur_material_indices, component_values)
            if write_dispersive:
                assert allowed_c1 is not None and allowed_c2 is not None and allowed_c3 is not None
                # allowed_cN[int_idx]: (*grid, num_poles, num_disp_components) -> (num_poles, num_disp_components, *grid)
                c1_slice = jnp.moveaxis(allowed_c1[int_idx], (-2, -1), (0, 1))
                c2_slice = jnp.moveaxis(allowed_c2[int_idx], (-2, -1), (0, 1))
                c3_slice = jnp.moveaxis(allowed_c3[int_idx], (-2, -1), (0, 1))
                if write_c4:
                    assert allowed_c4 is not None
                    c4_slice = jnp.moveaxis(allowed_c4[int_idx], (-2, -1), (0, 1))

        return _MaterialArraySlices(inv_perm_slice, c1_slice, c2_slice, c3_slice, c4_slice)


@autoinit
class EtchedDevice(Device):
    """A device that carves its region out of the existing background material.

    Instead of blending between two of its own materials, an etched device blends
    each voxel's *current* background permittivity with a single etch material
    (e.g. air), controlled by the continuous parameter value: ``0`` leaves the
    background untouched and ``1`` replaces it fully with the etch material. This
    models a partially etched structure carved out of whatever material already
    occupies the region (a substrate, a waveguide, ...).

    Because the result depends on the material already present, an etched device is
    :attr:`background_dependent` — ``apply_params`` restores the pristine
    background before every application.
    """

    @property
    def background_dependent(self) -> bool:
        return True

    def _validate_materials(self) -> None:
        if self.output_type != ParameterType.CONTINUOUS:
            raise Exception(
                f"Etched devices only support continuous parameters, {self.name} outputs {self.output_type}"
            )
        if len(self.materials) != 1:
            raise Exception(
                "Need exactly one material in etched device when parameter mapping outputs continuous "
                f"permittivity indices which replaces the eroded original materials, but got {self.materials}"
            )

    def _compute_material_slices(
        self,
        arrays: "ArrayContainer",
        params: dict[str, jax.Array] | jax.Array,
        **transform_kwargs,
    ) -> _MaterialArraySlices:
        if arrays.dispersive_c1 is not None:
            raise NotImplementedError(
                "Etched devices do not yet support dispersive materials. Use non-dispersive materials for the "
                "etch material and its background, or a standard Device."
            )
        # Single-channel device: __call__ returns a bare array.
        cur_material_indices = cast(jax.Array, self(params, expand_to_sim_grid=True, **transform_kwargs))

        num_components = arrays.inv_permittivities.shape[0]
        isotropic = num_components == 1
        diagonally_anisotropic = num_components == 3
        # (1, num_components) — an etched device has exactly one material.
        allowed_perm = jnp.asarray(
            compute_allowed_permittivities(
                self.materials,
                isotropic=isotropic,
                diagonally_anisotropic=diagonally_anisotropic,
            )
        )

        etch_perm = allowed_perm[0][:, None, None, None]  # (num_components, 1, 1, 1)
        background_perm = invert_property(arrays.inv_permittivities[:, *self.grid_slice])  # (num_components, *grid)
        perm_slice = background_perm + cur_material_indices * (etch_perm - background_perm)
        inv_perm_slice = invert_property(perm_slice)
        return _MaterialArraySlices(inv_perm_slice)


@autoinit
class PermittivityConductivityDevice(Device):
    """A device that optimizes permittivity and electric conductivity independently.

    Where the standard :class:`Device` derives every material property from a single
    weight per voxel — so a voxel is always some blend of two *whole* materials — this
    device carries two weights per voxel and interpolates the two properties separately:

    * the ``"permittivity"`` weight blends between the two materials' permittivities,
    * the ``"conductivity"`` weight blends between their electric conductivities.

    A weight pair of ``(1, 0)`` therefore produces the permittivity of the second
    material combined with the conductivity of the first — a combination that need not
    correspond to any real material in :attr:`materials`. That is the point: it lets a
    gradient-based optimizer explore refractive index and loss as independent degrees of
    freedom, which is useful for absorber and lossy-metasurface design.

    Both weights are interpolated along the *same* material ordering (ascending
    permittivity, see :func:`fdtdx.materials.compute_ordered_material_name_tuples`), so
    weight ``0`` always means "first material's value" for both properties. The two
    materials therefore bracket the reachable range of each property independently.

    Notes:
        The simulation only allocates a conductivity array when *some* material in the
        whole simulation is lossy. If neither of this device's materials has an
        ``electric_conductivity``, there is nothing to write into and the conductivity
        channel is inert — the device warns and optimizes permittivity alone.
    """

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("permittivity", "conductivity")

    def _validate_materials(self) -> None:
        if self.output_type != ParameterType.CONTINUOUS:
            raise Exception(
                f"{self.__class__.__name__} interpolates both material properties and therefore only supports "
                f"continuous parameters, but {self.name} outputs {self.output_type}"
            )
        if len(self.materials) != 2:
            raise Exception(
                f"Need exactly two materials in {self.__class__.__name__} to bracket the permittivity and "
                f"conductivity ranges, but got {self.materials}"
            )
        if not any(m.is_electrically_conductive for m in self.materials.values()):
            warnings.warn(
                f"{self.name} is a {self.__class__.__name__} but none of its materials {list(self.materials)} is "
                "electrically conductive, so the conductivity channel has no range to interpolate over and cannot "
                "affect the simulation. Give one of the materials a non-zero `electric_conductivity`, or use a "
                "standard Device if you only want to optimize permittivity.",
                UserWarning,
                stacklevel=2,
            )

    def _compute_material_slices(
        self,
        arrays: "ArrayContainer",
        params: dict[str, jax.Array] | jax.Array,
        **transform_kwargs,
    ) -> _MaterialArraySlices:
        if arrays.dispersive_c1 is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support dispersive materials: the conductivity channel and an "
                "ADE pole would both model loss, so their contributions could not be attributed. Use non-dispersive "
                "materials, or a standard Device if you need dispersion."
            )
        weights = self(params, expand_to_sim_grid=True, **transform_kwargs)
        assert isinstance(weights, dict)
        perm_weight = weights["permittivity"]
        cond_weight = weights["conductivity"]

        num_components = arrays.inv_permittivities.shape[0]
        # (2, num_components) — ordered ascending by permittivity.
        allowed_perm = jnp.asarray(
            compute_allowed_permittivities(
                self.materials,
                isotropic=num_components == 1,
                diagonally_anisotropic=num_components == 3,
            )
        )
        perm_bc = allowed_perm[:, :, None, None, None]  # (2, num_components, 1, 1, 1)
        perm_slice = perm_bc[0] + perm_weight * (perm_bc[1] - perm_bc[0])
        inv_perm_slice = invert_property(perm_slice)

        sigma_slice = None
        if arrays.electric_conductivity is not None:
            num_cond_components = arrays.electric_conductivity.shape[0]
            # Ordered by the same key as the permittivities, so index 0 is the same
            # material in both stacks and weight 0 consistently means "first material".
            allowed_sigma = jnp.asarray(
                compute_allowed_electric_conductivities(
                    self.materials,
                    isotropic=num_cond_components == 1,
                    diagonally_anisotropic=num_cond_components == 3,
                ),
                dtype=arrays.electric_conductivity.dtype,
            )
            sigma_bc = allowed_sigma[:, :, None, None, None]  # (2, num_cond_components, 1, 1, 1)
            sigma_slice = sigma_bc[0] + cond_weight * (sigma_bc[1] - sigma_bc[0])
            # Static objects are scaled the same way in _init_arrays; conductivity is
            # stored as the dimensionless update coefficient, not in S/m.
            sigma_slice = sigma_slice * self.conductivity_scale

        return _MaterialArraySlices(inv_perm_slice, electric_conductivity=sigma_slice)
