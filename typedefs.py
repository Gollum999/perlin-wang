from typing import Generic, NamedTuple, TypeVar


T = TypeVar('T')

class Rect(NamedTuple, Generic[T]):
    """ Holds four things, one for each side of a rectangle. """
    top: T
    right: T
    bottom: T
    left: T


ColorIdx = int
ColorEdge = list[int]  # a shuffled list of indices for looking up gradients on the edges of tiles

EdgeColors = Rect[ColorIdx]
ColorEdges = Rect[ColorEdge]
ColorToEdgeDict = dict[ColorIdx, ColorEdge]
