# `data_logging_FMI.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_FMI.py`  
**核心类：** `DataFMI`  
**功能定位：** 电成像（FMI）数据管理，支持多格式读取、空白条带删除、纹理特征计算、分形维数（FDE）计算、图像分割。  
**代码规模：** ~530 行（不含测试代码）

---

## 1. 类架构

### 1.1 异常类与枚举

```python
class FMIException(Exception)
class FileFormat(Enum):
    CSV = '.csv'
    TEXT = '.txt'
    IMAGE = '.image'
    UNKNOWN = 'unknown'
```

### 1.2 独立工具函数

#### `ele_stripes_delete(Pic, shape_target, delete_pix)`

**功能：** 删除 FMI 图像中的空白条带（无效像素），采用多退少补原则。

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `Pic` | `np.ndarray` | 原始 FMI 图像（二维） |
| `shape_target` | `Tuple[int, int]` | 目标形状 `(高度, 宽度)` |
| `delete_pix` | `float` | 像素删除阈值，`< delete_pix` 视为无效 |

**返回值：** `(处理后图像, 有效像素总数)`

**算法（逐行处理）：**

1. 遍历图像每一行
2. 查找 `!= delete_pix` 的有效像素索引
3. 根据有效像素数与目标宽度对比：
   - **相等**：直接使用
   - **不足**：用 0 填充
   - **过多**：用 `np.resize()` 压缩到目标宽度
4. 返回处理后的图像

---

### 1.3 核心类：`DataFMI`

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_data_fmi` | `np.ndarray` | 成像数据体（二维） |
| `_data_depth` | `np.ndarray` | 深度数据（一维） |
| `_resolution` | `float` | 默认 0.0025 米/点 |
| `_well_name` | `str` | 井名标识 |
| `path_fmi` | `str` | 数据文件路径 |
| `path_folder` | `str` | 文件所在目录 |
| `fmi_charter` | `str` | 仪器标识（DYNA/STAT/OTHERS） |
| `_is_data_loaded` | `bool` | 数据加载状态 |

#### 仪器标识自动推断

```python
if path_fmi.upper().__contains__('DYNA'):
    fmi_charter = 'DYNA'
elif path_fmi.upper().__contains__('STAT'):
    fmi_charter = 'STAT'
else:
    fmi_charter = 'OTHERS'
```

---

## 2. 核心方法

### 2.1 数据读取 `read_data()`

**支持三种文件格式：**

| 格式 | 读取方式 |
|------|----------|
| CSV | `pd.read_csv(index_col=0)` — 第一列为索引（深度） |
| TXT (LAS) | `np.loadtxt(skiprows=8)` — 跳过前 8 行头信息 |
| IMAGE (PNG/JPG/BMP) | `cv2.imread(GRAYSCALE)` + 从文件名解析深度 |

**图像文件的深度解析（特殊设计）：**

```python
# 从文件名路径中提取数字作为深度
# 例如: F:\...\3728.2-3729.2.bmp → 提取出 [3728.2, 3729.2]
depth_str_list = re.split(r'[/\\_-]', file_name)
depth_list = [float(s) for s in depth_str_list if float(s) succeeds]
# 用 linspace 生成等间隔深度
self._data_depth = np.linspace(min(depth_list[-2:]), max(depth_list[-2:]), rows)
```

### 2.2 空白条带删除 `ele_stripes_delete()`

**调用全局函数 `ele_stripes_delete()` 处理整幅图像。** 目标宽度按 `width_ratio=0.8` 计算。

```python
original_height, original_width = self._data_fmi.shape
target_width = int(original_width * width_ratio)
self._data_fmi, valid_count = ele_stripes_delete(...)
```

### 2.3 纹理特征计算 `get_texture()`

**调用外部模块：** `src_fmi.fmi_glcm_texture.cal_fmis_texture`

**GLCM 特征列表：**

| 缩写 | 特征名 | 说明 |
|------|--------|------|
| CON | Contrast | 对比度 |
| DIS | Dissimilarity | 相异度 |
| HOM | Homogeneity | 同质性 |
| ENG | Energy | 能量 |
| COR | Correlation | 相关性 |
| ASM | Angular Second Moment | 角二阶矩 |
| ENT | Entropy | 熵 |

**统计方向（4 种）：** `MEAN`, `SUB`, `X`, `Y`  
**生成的列名格式：** `{FEATURE}_{DIRECTION}_{FMI_CHARTER}`  
例如：`CON_MEAN_STAT`, `HOM_X_DYNA`

**缓存机制：**

```python
if os.path.exists(fmi_texture_path):
    return pd.read_csv(fmi_texture_path)  # 直接读取已缓存文件
```

**默认配置：**

```python
{
    'level': 16,           # 灰度级别
    'distance': [2, 4],    # 像素距离
    'angles': [0, π/2],    # 0° 和 90° 两个方向
    'windows_length': 200, # 滑动窗口长度（点数）
    'windows_step': 100    # 滑动步长
}
```

### 2.4 分形维数计算 `get_fmi_fde()`

**调用外部模块：**

- `src_fmi.fmi_fractal_dimension_extended_calculate.cal_fmis_fractal_dimension_extended`
- `src_fmi.fmi_fractal_dimension_extended_calculate.trans_NMR_as_Ciflog_file_type`

**流程：**

1. 检查缓存文件是否存在 → 直接读取
2. 否则调用 `cal_fmis_fractal_dimension_extended()` 计算
3. 转换格式后保存到 `_fde_NMR.txt`
4. 返回分形谱矩阵 `(depth, fde_values)`

**缓存文件格式：** 制表符分隔的 TXT，`fmt='%.4f'`

### 2.5 图像分割 `fmi_segment()`

**调用外部模块：** `src_fmi.fmi_segmentation.cal_fmis_segmentation`

**支持 6 种分割方法：**

| 方法标识 | 方法名 | 说明 |
|----------|--------|------|
| `otsu` | Otsu Threshold | Otsu 大津阈值 |
| `tophat_otsu` | TopHat + Otsu | 顶帽变换 + Otsu |
| `adaptive` | Adaptive Threshold | 自适应阈值 |
| `kmeans` | K-means (K=3) | K 均值聚类 |
| `gmm` | GMM (n=3) | 高斯混合模型 |
| `wavelet` | Wavelet (db4) | 小波变换 |

**输出：** 每种方法返回分割后的图像数据，并保存为 `_{method}_seg.txt`

**保存格式（含 LAS 头信息）：**

```
WELLNAME= {well_name}_{fmi_charter}_{method}
STDEP   = {depth_start}
ENDEP   = {depth_end}
LEV     = 0.0025
UNIT    = meter
CURNAMES= IMAGE.DYNA

DEPTH
```

### 2.6 图像预处理 `fmi_preprocess()`

预留接口，支持 4 种预处理方式：

| 方式 | 说明 |
|------|------|
| `gabor` | Gabor 滤波器 |
| `gama` | Gamma 变换 |
| `wavelet` | 小波变换 |
| `filter` | 保边滤波器（导向滤波/双边滤波/最小二乘滤波） |

**当前版本：** 仅预留接口，未实现具体逻辑。

---

## 3. 数据流图

```
原始文件 (CSV/TXT/IMAGE)
         │
         ▼
   read_data()
         │
         ├─► CSV: pd.read_csv(index_col=0)
         ├─► TXT: np.loadtxt(skiprows=8)
         └─► IMAGE: cv2.imread + 深度解析
         │
         ▼
   _data_fmi (成像数据) + _data_depth (深度数据)
         │
         ├──► ele_stripes_delete() → 空白条带删除
         ├──► get_texture() → GLCM 纹理特征
         ├──► get_fmi_fde() → 分形维数谱
         └──► fmi_segment() → 图像分割
```

---

## 4. 设计亮点

| 特性 | 实现方式 |
|------|----------|
| 多格式支持 | CSV/TXT/IMAGE 三种格式自动识别 |
| 惰性计算 | 纹理/FDE/分割结果按需计算，已存在则直接读取 |
| 深度自动解析 | 从图像文件名提取深度信息 |
| 仪器标识自动推断 | 从路径关键字推断 DYNA/STAT |
| 缓存持久化 | 所有计算结果保存到文件目录 |

---

## 5. 潜在问题

1. **`fmi_preprocess()` 方法未实现**：预留了参数和框架，但 Gabor/Gamma/Wavelet/Filter 均未具体实现
2. **`get_texture()` 中 `level=16` 灰度级别可能导致信息损失**（FMI 原始数据 0-255）
3. **图像文件的深度解析依赖文件名格式**，若格式不规则可能解析失败

---

*文档版本：1.0 | 对应源码版本：2026-06-19*
