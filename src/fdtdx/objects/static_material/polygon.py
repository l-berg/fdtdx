import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.grid import polygon_to_mask
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class ExtrudedPolygon(StaticMultiMaterialObject):
    """A polygon object specified by a list of vertices. The coordinate system has its origin at the lower left of the
    bounding box of the polygon.

    Attributes:
        material_name (str): Name of the material in the materials dictionary to be used for the object
        axis (int): The extrusion axis.
        vertices (np.ndarray): numpy array of shape (N, 2) specifying the position of vertices in metrical units
            (meter).
    """

    material_name: str = frozen_field()
    axis: int = frozen_field()
    vertices: np.ndarray = frozen_field()

    @property
    def horizontal_axis(self) -> int:
        """Gets the horizontal axis perpendicular to the fiber axis.

        Returns:
            int: The index of the horizontal axis (0=x or 1=y).
        """
        return 1 if self.axis == 0 else 0

    @property
    def vertical_axis(self) -> int:
        """Gets the vertical axis perpendicular to the fiber axis.

        Returns:
            int: The index of the vertical axis (1=y or 2=z).
        """
        return 1 if self.axis == 2 else 2

    @property
    def centered_vertices(self) -> np.ndarray:
        vx = self.vertices[:, 0] + 0.5 * self.real_shape[self.horizontal_axis]
        vy = self.vertices[:, 1] + 0.5 * self.real_shape[self.vertical_axis]
        return np.stack((vx, vy), axis=-1)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        n_horizontal = self.grid_shape[self.horizontal_axis]
        n_vertical = self.grid_shape[self.vertical_axis]

        half_res = 0.5 * self._config.resolution
        max_horizontal = (n_horizontal - 0.5) * self._config.resolution
        max_vertical = (n_vertical - 0.5) * self._config.resolution

        # move vertices to center

        #mask_2d_orig = polygon_to_mask( boundary=(half_res, half_res, max_horizontal, max_vertical), resolution=self._config.resolution, polygon_vertices=self.centered_vertices,)
        mask_2d = polygon_to_mask_hybrid(
        #mask_2d = polygon_to_mask_pure_shapely(
        #mask_2d = polygon_to_mask(
            boundary=(half_res, half_res, max_horizontal, max_vertical),
            resolution=self._config.resolution,
            polygon_vertices=self.centered_vertices,
        )
        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(mask_2d, axis=self.axis),
            #jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )

        return mask

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr

import shapely
from matplotlib.path import Path
from scipy.ndimage import binary_dilation, binary_erosion

def polygon_to_mask_pure_shapely(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_vertices: np.ndarray,
) -> np.ndarray:
    """The Shapely-based fractional area mask function."""
    min_x, min_y, max_x, max_y = boundary
    x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    
    half_res = resolution / 2.0
    #boxes = shapely.box(X.ravel(), Y.ravel(), (X + resolution).ravel(), (Y + resolution).ravel())
    boxes = shapely.box((X - half_res).ravel(), (Y - half_res).ravel(), 
                        (X + half_res).ravel(), (Y + half_res).ravel())
    poly = shapely.Polygon(polygon_vertices)
    
    intersections = shapely.intersection(boxes, poly)
    areas = shapely.area(intersections)
    
    return (areas / (resolution ** 2)).reshape(X.shape)

def polygon_to_mask_hybrid(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_vertices: np.ndarray,
    safe_margin: int = 2
) -> np.ndarray:
    """
    Generate a 2D fractional mask, highly optimized for large grids.
    Mixes fast binary point-in-polygon with exact Shapely boundary intersections.
    """
    assert polygon_vertices.ndim == 2
    assert polygon_vertices.shape[1] == 2
    min_x, min_y, max_x, max_y = boundary

    # 1. Create coordinate arrays
    x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    # 2. Fast Binary Masking (Core Interior/Exterior)
    points = np.column_stack((X.ravel(), Y.ravel()))
    polygon_path = Path(polygon_vertices)
    inside_polygon = polygon_path.contains_points(points, radius=1e-3*resolution)
    binary_mask = inside_polygon.reshape(X.shape)

    # 3. Identify the Boundary Band using Scipy Morphology
    # Dilation pushes the boundary out, erosion pulls it in. 
    # The difference between them is our "uncertainty band" where edges lie.
    dilated = binary_dilation(binary_mask, iterations=safe_margin)
    eroded = binary_erosion(binary_mask, iterations=safe_margin)
    boundary_mask = dilated ^ eroded  # XOR gives us only the boundary shell

    # Initialize the output float array with our fast binary results
    fractional_mask = binary_mask.astype(float)

    # Get the 2D array indices of the boundary pixels
    rows, cols = np.where(boundary_mask)

    # 4. Run exact Shapely calculation ONLY on the boundary pixels
    if len(rows) > 0:
        # Extract physical coordinates for just the boundary pixels
        cx = X[rows, cols]
        cy = Y[rows, cols]
        
        half_res = resolution / 2.0
        
        # Create vectorized Shapely boxes for the boundary pixels
        boxes = shapely.box(
            cx - half_res, cy - half_res, 
            cx + half_res, cy + half_res
        )
        
        poly = shapely.Polygon(polygon_vertices)
        
        # Calculate exact intersections
        intersections = shapely.intersection(boxes, poly)
        areas = shapely.area(intersections)
        
        # Overwrite the boundary band with exact fractions (0.0 to 1.0)
        pixel_area = resolution ** 2
        fractional_mask[rows, cols] = areas / pixel_area

    return fractional_mask
