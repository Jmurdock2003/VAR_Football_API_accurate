# bbox_utils.py: Utility functions for bounding box operations and geometry calculations

def get_bbox_area(bbox):
    """
    Compute the area of a bounding box.

    Parameters:
      bbox (list[float]): [x1, y1, x2, y2]

    Returns:
      float: Area = width * height
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def get_centre(bbox):
    """
    Calculate the center point of a bounding box.

    Parameters:
      bbox (list[float]): [x1, y1, x2, y2]

    Returns:
      list[float]: [cx, cy] center coordinates
    """
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def get_bbox_width(bbox):
    """
    Retrieve the width of a bounding box.

    Parameters:
      bbox (list[float]): [x1, y1, x2, y2]

    Returns:
      float: Width = x2 - x1
    """
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """
    Compute Euclidean distance between two points.

    Parameters:
      p1 (tuple[float, float]): (x, y)
      p2 (tuple[float, float]): (x, y)

    Returns:
      float: sqrt((x1 - x2)^2 + (y1 - y2)^2)
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    """
    Compute component-wise distance between two points.

    Parameters:
      p1 (tuple[float, float]): (x, y)
      p2 (tuple[float, float]): (x, y)

    Returns:
      tuple[float, float]: (dx, dy)
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    """
    Approximate the foot contact position (bottom-center of bbox).

    Parameters:
      bbox (list[float]): [x1, y1, x2, y2]

    Returns:
      tuple[int, int]: (cx, y2) integer coordinates
    """
    x1, y1, x2, y2 = bbox
    # Bottom center: average x, max y
    return int((x1 + x2) / 2), int(y2)
