这段代码简单模拟了使用高斯溅射将z=0平面上的两个点投射到空间中的过程（不过受限于数学能力，我还是不大能完全理解高斯核的工作原理）

```python
import torch

def gaussian_kernel(x, y, z, mu, sigma):
    """
    高斯核函数
    :param x, y, z: 3D网格坐标
    :param mu: 均值 (中心点)
    :param sigma: 标准差 (控制高斯核的扩散范围)
    :return: 高斯核值
    """
    diff = torch.stack([x - mu[0], y - mu[1], z - mu[2]], dim=-1)
    exponent = -0.5 * torch.sum((diff / sigma)**2, dim=-1)
    return torch.exp(exponent)
```


```python
def generate_3d_grid(x_range, y_range, z_range, resolution):
    """
    生成3D网格
    :param x_range, y_range, z_range: 空间范围 (min, max)
    :param resolution: 网格分辨率
    :return: 3D网格坐标
    """
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    z = torch.linspace(z_range[0], z_range[1], resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    return grid_x, grid_y, grid_z
```


```python
def gaussian_splatting_3d(points, intensities, grid_x, grid_y, grid_z, sigma):
    """
    高斯溅射到3D网格
    :param points: 二维点集 [N, 2]
    :param intensities: 点的强度值 [N]
    :param grid_x, grid_y, grid_z: 3D网格坐标
    :param sigma: 高斯核标准差
    :return: 3D密度分布
    """
    density = torch.zeros_like(grid_x)

    for i, point in enumerate(points):
        mu = torch.tensor([point[0], point[1], 0.0])  # 假设z=0
        kernel = gaussian_kernel(grid_x, grid_y, grid_z, mu, sigma)
        density += intensities[i] * kernel

    return density


```


```python
import matplotlib.pyplot as plt

def visualize_3d(density):
    """
    简单3D可视化
    :param density: 三维密度分布
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(density > density.mean())  # 使用密度平均值作为阈值
    plt.show()
```


```python
# 参数设置
points = torch.tensor([[0.5, 0.5], [1.5, 1.5]])  # 二维点
intensities = torch.tensor([1.0, 0.8])  # 点的强度
x_range, y_range, z_range = (0, 2), (0, 2), (0, 2)
resolution = 50
sigma = torch.tensor([0.1, 0.1, 0.1])  # 高斯核标准差

# 生成网格
grid_x, grid_y, grid_z = generate_3d_grid(x_range, y_range, z_range, resolution)

# 高斯溅射
density = gaussian_splatting_3d(points, intensities, grid_x, grid_y, grid_z, sigma)

# 可视化
visualize_3d(density)
```


    
![png](output_4_0.png)
    



```python

```
