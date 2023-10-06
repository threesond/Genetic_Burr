import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from utils import array_to_string, generate_base_shape, string_to_array, validate_shape

### first we generate base pieces, that is, the valid piece with minimum voxels(valid means the piece should be internally connected)
### when the puzzle getting more complex, it's harder to deduce the shape of the base pieces
### so we generate them automatically


voxel_string = '#+++++##+++++##+++++##+++++##+++++##+++++##+++++##+++++#'
voxel_array = string_to_array(voxel_string, (2,4,7))

while True:
    base_shape_candidate = generate_base_shape(voxel_array, 5)
    validation = validate_shape(base_shape_candidate)
    if validation:
        print(base_shape_candidate)
        voxel_string = array_to_string(base_shape_candidate)
        print(voxel_string)
        break