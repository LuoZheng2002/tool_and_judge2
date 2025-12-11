from typing import overload, Literal, Union, Any

class Shape:
    """
    Base class for Shape enum — actual instances will have a concrete variant type.
    """
    # so that one can write Shape.Circle instead of Shape_Circle
    Circle = Shape_Circle
    Rectangle = Shape_Rectangle
    RegularPolygon = Shape_RegularPolygon
    Nothing = Shape_Nothing
    pass

class Shape_Circle(Shape):
    def __new__(cls, radius: float = 1.0) -> Shape: ...
    @property
    def radius(self) -> float: ...

class Shape_Rectangle(Shape):
    def __new__(cls, *, width: float, height: float) -> Shape: ...
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...

class Shape_RegularPolygon(Shape):
    def __new__(cls, side_count: int, radius: float = 1.0) -> Shape: ...
    @property
    def side_count(self) -> int: ...
    @property
    def radius(self) -> float: ...

class Shape_Nothing(Shape):
    def __new__(cls) -> Shape: ...


