from collections.abc import Callable
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__ = "0.1.0"


class RegularGrid:
    """规则网格"""

    def __init__(
        self,
        shape: tuple[int, int],
        extents: tuple[float, float, float, float] = (0, 1, 0, 1),
        origin: Literal["upper", "lower"] = "upper",
        align_corners: bool = False,
    ) -> None:
        """用网格行列数 shape 和边界框 extents 初始化网格"""
        self.shape = shape
        self.extents = extents
        self.origin = origin
        self.align_corners = align_corners

        self.ny, self.nx = self.shape
        self.x0, self.x1, self.y0, self.y1 = self.extents

        if self.nx < 1 or self.ny < 1:
            raise ValueError("要求 nx >= 1 且 ny >= 1")
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError("要求 x0 <= x1 且 y0 <= y1")

        if self.align_corners:
            self.dx = (self.x1 - self.x0) / max(self.nx - 1, 1)
            self.dy = (self.y1 - self.y0) / max(self.ny - 1, 1)
            self.x = np.linspace(self.x0, self.x1, self.nx)
            self.y = np.linspace(self.y0, self.y1, self.ny)
        else:
            self.dx = (self.x1 - self.x0) / self.nx
            self.dy = (self.y1 - self.y0) / self.ny
            self.x = (np.arange(self.nx) + 0.5) * self.dx + self.x0
            self.y = (np.arange(self.ny) + 0.5) * self.dy + self.y0

        if self.origin == "upper":
            self.y = self.y[::-1]

    def __repr__(self) -> str:
        kwargs = [
            f"{self.shape=}",
            f"{self.extents=}",
            f"{self.origin=}",
            f"{self.align_corners=}",
        ]
        kwargs = ", ".join(kwargs).replace("self.", "")
        info = f"{self.__class__.__name__}({kwargs})"

        return info

    @property
    def X(self) -> NDArray[np.float64]:
        return np.broadcast_to(self.x, self.shape)

    @property
    def Y(self) -> NDArray[np.float64]:
        return np.broadcast_to(self.y[:, np.newaxis], self.shape)

    def _round(self, a: ArrayLike) -> NDArray[np.int32]:
        a = np.asarray(a)
        if self.align_corners:
            a = np.rint(a)
        return a.astype(int)

    def row_index(self, y: ArrayLike) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
        """获取纵坐标 y 在网格里对应的行索引"""
        y = np.asarray(y)
        if self.dy > 0:
            index = (y - self.y0) / self.dy
            index = self._round(index)
            if self.origin == "upper":
                index = self.ny - 1 - index
            index = index.clip(0, self.ny - 1)
        else:
            index = np.zeros_like(y, dtype=int)

        inside = (y >= self.y0) & (y <= self.y1)

        return index, inside

    def col_index(self, x: ArrayLike) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
        """获取横坐标 x 在网格里对应的列索引"""
        x = np.asarray(x)
        if self.dx > 0:
            index = (x - self.x0) / self.dx
            index = self._round(index)
            index = index.clip(0, self.nx - 1)
        else:
            index = np.zeros_like(x, dtype=int)

        inside = (x >= self.x0) & (x <= self.x1)

        return index, inside

    def indices(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[tuple[NDArray[np.int32], NDArray[np.int32]], NDArray[np.bool_]]:
        """获取横坐标 x 和纵坐标 y 在网格里对应的行列索引"""
        row_index, row_inside = self.row_index(y)
        col_index, col_inside = self.col_index(x)
        indices = (row_index, col_index)
        inside = row_inside & col_inside

        return indices, inside


def transform_extent(
    extents: tuple[float, float, float, float], transform: Callable, npts: int = 20
) -> tuple[float, float, float, float]:
    """将 extents 插值成 4 * npts 个点, 应用 transform 变换后得到新的 extents。"""
    x0, x1, y0, y1 = extents
    xp = np.array([x0, x1, x1, x0, x0])
    yp = np.array([y0, y0, y1, y1, y0])
    ip = np.arange(5)
    i = np.linspace(0, 4, 4 * npts + 1)
    x = np.interp(i, ip, xp)
    y = np.interp(i, ip, yp)
    x, y = transform(x, y)

    return x.min(), x.max(), y.min(), y.max()


def reproject(
    grid1: RegularGrid,
    grid2: RegularGrid,
    inv_transform: Optional[Callable] = None,
) -> tuple[tuple[NDArray[np.int32], NDArray[np.int32]], NDArray[np.bool_]]:
    """用最近邻插值将 grid1 重投影到 grid2，得到用来索引 grid1 上变量的数组。"""
    X, Y = grid2.X, grid2.Y
    if inv_transform is not None:
        X, Y = inv_transform(X, Y)
    return grid1.indices(X, Y)


def resize(
    shape1: tuple[int, int], shape2: tuple[int, int]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """用最近邻插值将形如 shape1 的图片 resize 至 shape2"""
    grid1 = RegularGrid(shape1)
    grid2 = RegularGrid(shape2)
    indices, _ = reproject(grid1, grid2)

    return indices
