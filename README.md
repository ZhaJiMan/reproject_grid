# reproject_grid

## 简介

用最近邻插值将坐标系 1 中的规则网格重投影成坐标系 2 中的规则网格。

规则网格定义为横坐标等距分布，纵坐标等距分布，横纵坐标轴相互垂直的网格。

## 算法

- 定义坐标系 1 中的网格 1 和坐标系 2 中的网格 2
- 定义坐标系之间的变换，将网格 2 中所有网格点坐标变换到坐标系 1 中。
- 计算这些点在网格 1 中落入到哪些格子里，得到格子的行列数组。
- 用行列数组去索引基于网格 1 的变量数组，相当于将变量从网格 1 最近邻插值到了网格 2 上。
- 如果网格 2 中的点在网格 1 中出界了，将界外值设为缺测。

## 依赖

```
python>=3.9
numpy>=1.20.0
```

## 用法

定义规则网格后获取网格坐标：

```python
from reproject_grid import RegularGrid

grid = RegularGrid((10, 10), (70, 140, 0, 60))
print(grid.x0, grid.x1, grid.y0, grid.y1)
print(grid.x, grid.y)
print(grid.X, grid.Y)
```

找出坐标点落入规则网格中第几行第几列的格子：

```python
row, inside = grid.row_index(40)
if inside:
    print('row': i)

col, inside = grid.col_index(115)
if inside:
    print('col': col)
```

将等经纬度投影的图片数组 `Z1` 重投影成网络墨卡托投影的图片：

```python
from functools import partial

import numpy as np
from pyproj import Proj
from reproject_grid import RegularGrid, reproject, transform_extent

proj = Proj("+proj=webmerc")
transform = proj
inv_transform = partial(proj, inverse=True)

shape = (1000, 1000)
grid1 = RegularGrid(shape, (70, 140, 0, 60))
grid2 = RegularGrid(shape, transform_extent(grid1.extents, transform))
indices, inside = reproject(grid1, grid2, inv_transform)

Z1 = grid1.X + grid1.Y
Z2 = Z1[indices]
Z2[~inside] = np.nan
```

完整例子和图示见：[example.ipynb](example.ipynb)

## 其它

更泛用的重投影请见：

- [scipy.spatial.KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html)
- [gdal.Warp](https://gdal.org/en/latest/api/python/utilities.html)
- [Pyresample](https://github.com/pytroll/pyresample)

## 参考连接

[Pillow vs cv2 resize #2718](https://github.com/python-pillow/Pillow/issues/2718)

[Where Are Pixels? -- a Deep Learning Perspective](https://ppwwyyxx.com/blog/2021/Where-are-Pixels/)
