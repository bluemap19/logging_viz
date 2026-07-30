# `data_logging_NMR.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_NMR.py`  
**核心类：** `DataNMR`  
**功能定位：** 核磁共振（NMR）T2 谱数据管理，支持多格式读取和深度筛选。  
**代码规模：** ~220 行（不含测试代码）

---

## 1. 类架构

### 1.1 异常类与枚举

```python
class NMRException(Exception)
class FileFormat(Enum):
    CSV = '.csv'
    TEXT = '.txt'
    UNKNOWN = 'unknown'
```

### 1.2 核心类：`DataNMR`

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_table_2` | `pd.DataFrame` | 保留属性（用于兼容性） |
| `_data_nmr` | `np.ndarray` | NMR T2 谱数据体（二维） |
| `_data_depth` | `np.ndarray` | 深度数据（一维） |
| `_resolution` | `float` | 默认 0.0025 米/点 |
| `_well_name` | `str` | 井名标识 |
| `path_nmr` | `str` | 数据文件路径 |
| `nmr_charter` | `str` | 仪器标识（NMR/UNKNOWN） |
| `_is_data_loaded` | `bool` | 数据加载状态 |

#### 仪器标识自动推断

```python
if path_nmr.upper().__contains__('NMR'):
    nmr_charter = 'NMR'
else:
    nmr_charter = 'UNKNOWN'
```

---

## 2. 核心方法

### 2.1 数据读取 `read_data()`

**支持两种文件格式：**

| 格式 | 读取方式 |
|------|----------|
| CSV | `pd.read_csv(index_col=0)` — 第一列为索引（深度） |
| TXT | `np.loadtxt(delimiter='\t', skiprows=8)` — 制表符分隔，跳过前 8 行 |

**数据结构：**

- `_data_nmr`：每一行对应一个深度点的 T2 谱（每个元素是一个 T2 时间的信号幅度）
- `_data_depth`：与行数对应的一维深度数组

### 2.2 数据获取 `get_data()`

```python
def get_data(depth: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]
```

**参数：**

- `depth=None`：返回全部数据
- `depth=[min, max]`：返回指定深度范围内的数据

**返回：** `(nmr_data, depth_data)` 元组

### 2.3 数据摘要 `get_summary()`

返回字典，包含：

| 字段 | 说明 |
|------|------|
| `well_name` | 井名 |
| `nmr_charter` | 仪器标识 |
| `file_path` | 文件路径 |
| `is_loaded` | 加载状态 |
| `resolution` | 分辨率 |
| `data_shape` | NMR 数据形状 |
| `depth_range` | 深度范围 (min, max) |
| `data_type` | NumPy 数据类型 |

### 2.4 详细信息 `get_data_info()`

返回格式化的多行字符串：

```
井名: XXX
仪器: NMR
数据形状: (rows, cols)
深度范围: 1000.00 - 1500.00
数据类型: float64
分辨率: 0.0025 米/点
```

---

## 3. 与 `DataFMI` 的对比

| 对比维度 | DataFMI | DataNMR |
|----------|---------|---------|
| 数据维度 | 二维图像（行=深度，列=电极） | 二维谱（行=深度，列=T2 时间） |
| 高级处理 | 纹理/FDE/分割（未实现预处理） | 无高级处理模块 |
| 缓存机制 | 有（纹理/FDE 文件） | 无 |
| 空白条带删除 | 有 | 无 |
| 预处理接口 | 有（未实现） | 无 |

---

## 4. 潜在问题

1. **功能相对简单**：与 `DataFMI` 相比，`DataNMR` 缺少纹理计算、分形维数、分割等高级处理功能
2. **无缓存机制**：每次调用 `get_data()` 都会重新从文件读取
3. **`_table_2` 属性未使用**：已声明但从未被赋值或调用
4. **TXT 文件格式假设过于固定**：`skiprows=8` 和 `delimiter='\t'` 均为硬编码

---

*文档版本：1.0 | 对应源码版本：2026-06-19*
