import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform


@autoinit
class DiagonalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce diagonal symmetry by effectively halving the parameter space.
    The symmetry is achieved by transposing the image and taking the mean
    of the original and transpose.

    This creates a design that is symmetric across one of the two diagonals.
    """

    #: If true, the symmetry axis is from (x_min, y_min) to (x_max, y_max).
    #: If false, the other diagonal (from (x_min, y_max) to (x_max, y_min)) is used.
    min_min_to_max_max: bool = frozen_field()

    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # convert to 2d
            vertical_axis = v.shape.index(1)
            v_2d = v.squeeze(vertical_axis)

            # enforce symmetry
            if self.min_min_to_max_max:
                other = v_2d.T
            else:
                other = v_2d[::-1, ::-1].T
            cur_mean = (v_2d + other) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result


@autoinit
class HorizontalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce horizontal (x-axis) mirror symmetry.

    This creates a design that is symmetric across a vertical line through
    the center, i.e., the left half mirrors the right half.
    The symmetry is enforced by averaging the array with its horizontally
    flipped version.
    """

    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # convert to 2d
            vertical_axis = v.shape.index(1)
            v_2d = v.squeeze(vertical_axis)

            # enforce symmetry: flip along x-axis (axis 0)
            flipped = v_2d[::-1, :]
            cur_mean = (v_2d + flipped) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result


@autoinit
class VerticalSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce vertical (y-axis) mirror symmetry.

    This creates a design that is symmetric across a horizontal line through
    the center, i.e., the top half mirrors the bottom half.
    The symmetry is enforced by averaging the array with its vertically
    flipped version.
    """

    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # convert to 2d
            vertical_axis = v.shape.index(1)
            v_2d = v.squeeze(vertical_axis)

            # enforce symmetry: flip along y-axis (axis 1)
            flipped = v_2d[:, ::-1]
            cur_mean = (v_2d + flipped) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result


@autoinit
class PointSymmetry2D(SameShapeTypeParameterTransform):
    """
    Enforce 180-degree rotational (point) symmetry.

    This creates a design that is symmetric under 180-degree rotation about
    its center. The symmetry is enforced by averaging the array with its
    180-degree rotated version.
    """

    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # convert to 2d
            vertical_axis = v.shape.index(1)
            v_2d = v.squeeze(vertical_axis)

            # enforce symmetry: 180-degree rotation (flip both axes)
            rotated = v_2d[::-1, ::-1]
            cur_mean = (v_2d + rotated) / 2

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result

@autoinit
class RotationalSymmetry90_2D(SameShapeTypeParameterTransform):
    """
    Enforce 90-degree (four-fold) rotational symmetry.

    This creates a design that is symmetric under 90-degree rotation about
    its center. The symmetry is enforced by averaging the array with its
    90, 180, and 270-degree rotated versions. 
    
    Note: The 2D dimensions of the array must be square.
    """

    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            # convert to 2d
            vertical_axis = v.shape.index(1)
            v_2d = v.squeeze(vertical_axis)

            # enforce symmetry: average over 0, 90, 180, and 270 degrees
            rot90 = jnp.rot90(v_2d, k=1)
            rot180 = jnp.rot90(v_2d, k=2)
            rot270 = jnp.rot90(v_2d, k=3)
            
            cur_mean = (v_2d + rot90 + rot180 + rot270) / 4.0

            # expand dims again
            result[k] = jnp.expand_dims(cur_mean, vertical_axis)
        return result
