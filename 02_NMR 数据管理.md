# 🧲 NMR 数据管理模块

> NMR (Nuclear Magnetic Resonance) Data Management Module

**版本：** v1.0  
**最后更新：** 2026-03-14  
**关联源文件：** `src_well_data/data_logging_NMR.py`

---

## 一、模块概述

### 1.1 模块定位

NMR 数据管理模块专门负责**核磁共振测井数据**的加载、处理和可视化准备。核磁共振测井通过测量地层中氢核的弛豫特性，获取孔隙度、孔径分布、流体类型等关键信息，是储层评价的重要工具。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| **数据加载** | 读取 CSV 格式的 NMR T2 谱数据 |
| **数据解析** | 解析多深度点的 T2 弛豫谱 |
| **数据插值** | 支持深度对齐和插值处理 |
| **数据封装** | 统一的数据访问接口 |

### 1.3 技术特点

- 支持多深度点 T2 谱数据
- T2 弛豫时间范围覆盖 0.1ms - 1000ms
- 支持对数坐标和线性坐标显示
- 可与常规测井数据深度对齐

### 1.4 应用场景

- 孔隙度计算
- 孔径分布分析
- 可动流体与束缚流体划分
- 渗透率估算
- 流体类型识别

---

## 二、文件结构与格式

### 2.1 源文件组织

```
src_well_data/
└── data_logging_NMR.py        # DataNMR 类定义
```

### 2.2 数据文件格式

#### NMR T2 谱数据 (CSV)

**命名规则：** 包含 `nmr` 关键字，如 `{井名}_nmr_data.csv`

**文件结构：**
| 列名 | 类型 | 说明 |
|------|------|------|
| DEPTH | float | 深度 (m) |
| T2_0.1 | float | T2=0.1ms 处的振幅 |
| T2_0.2 | float | T2=0.2ms 处的振幅 |
| T2_0.3 | float | T2=0.3ms 处的振幅 |
| ... | ... | ... |
| T2_1000 | float | T2=1000ms 处的振幅 |

**示例数据：**
```csv
DEPTH,T2_0.1,T2_0.2,T2_0.3,T2_0.5,T2_1.0,...,T2_1000
3200.5,0.05,0.12,0.25,0.48,0.89,...,0.02
3200.6,0.06,0.14,0.28,0.52,0.95,...,0.03
3200.7,0.04,0.10,0.22,0.45,0.82,...,0.01
```

**数据特点：**
- 第一列为深度值 (单位：米)
- 后续列为不同 T2 弛豫时间处的振幅值
- T2 时间通常按对数间隔分布 (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, ..., 1000 ms)
- 振幅值反映对应弛豫时间的孔隙流体含量

---

## 三、DataNMR 类设计

### 3.1 类结构

```python
class DataNMR:
    """NMR 数据管理类"""
    
    def __init__(self, path: str, well_name: str):
        """
        初始化 NMR 数据对象
        
        参数:
            path: NMR 数据文件路径
            well_name: 井名
        """
        self.path = path
        self.well_name = well_name
        self.data = None
        self.depth = None
        self.t2_times = None
        self.spectra = None
    
    def load_data(self) -> None:
        """加载 NMR 数据文件"""
        pass
    
    def get_data(self, depth: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取 NMR 数据
        
        参数:
            depth: 可选的深度范围限制
        
        返回:
            (depth_array, t2_times, spectra) 元组
        """
        pass
    
    def get_t2_spectrum_at_depth(self, target_depth: float, tolerance: float = 0.1) -> Optional[np.ndarray]:
        """
        获取指定深度的 T2 谱
        
        参数:
            target_depth: 目标深度
            tolerance: 深度容差
        
        返回:
            T2 谱数组或 None
        """
        pass
    
    def calculate_porosity(self, depth_range: List[float]) -> float:
        """
        计算指定深度范围内的总孔隙度
        
        参数:
            depth_range: [depth_min, depth_max]
        
        返回:
            平均孔隙度值
        """
        pass
    
    def calculate_ffb_bfi(self, depth_range: List[float], t2_cutoff: float = 100.0) -> Tuple[float, float]:
        """
        计算自由流体和束缚流体指数
        
        参数:
            depth_range: [depth_min, depth_max]
            t2_cutoff: T2 截止值 (ms)，默认 100ms
        
        返回:
            (FFI, BFI) 元组
        """
        pass
```

### 3.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `path` | str | NMR 数据文件路径 |
| `well_name` | str | 所属井名 |
| `data` | pd.DataFrame | 原始数据 DataFrame |
| `depth` | np.ndarray | 深度数组 |
| `t2_times` | np.ndarray | T2 弛豫时间数组 (ms) |
| `spectra` | np.ndarray | T2 谱数据矩阵 (深度×T2 时间) |
| `porosity` | np.ndarray | 计算的孔隙度数组 (缓存) |

### 3.3 核心方法详解

#### load_data()

**目的：** 从 CSV 文件加载 NMR T2 谱数据

**原理：**
1. 使用 `pandas.read_csv()` 读取文件
2. 提取第一列作为深度数组
3. 提取 T2_*列名，解析出 T2 弛豫时间值
4. 将谱数据转换为 numpy 数组

**代码示例：**
```python
def load_data(self):
    import pandas as pd
    import numpy as np
    import re
    
    # 读取 CSV
    self.data = pd.read_csv(self.path)
    
    # 提取深度
    self.depth = self.data['DEPTH'].values
    
    # 提取 T2 列名并解析 T2 时间
    t2_columns = [col for col in self.data.columns if col.startswith('T2_')]
    self.t2_times = np.array([float(re.search(r'T2_(\d+\.?\d*)', col).group(1)) 
                              for col in t2_columns])
    
    # 提取谱数据
    self.spectra = self.data[t2_columns].values
```

#### get_data(depth)

**目的：** 获取指定深度范围的 NMR 数据

**参数：**
- `depth`: 可选，深度范围 `[depth_min, depth_max]`

**返回：**
- `depth_array`: 深度数组
- `t2_times`: T2 弛豫时间数组
- `spectra`: T2 谱数据矩阵

**原理：**
1. 如果未指定 depth，返回全部数据
2. 如果指定 depth，进行深度范围筛选
3. 返回筛选后的数据

#### get_t2_spectrum_at_depth(target_depth, tolerance)

**目的：** 获取最接近指定深度的 T2 谱

**参数：**
- `target_depth`: 目标深度值
- `tolerance`: 深度容差 (默认 0.1m)

**返回：**
- 最接近目标深度的 T2 谱数组，如果无匹配则返回 None

**应用场景：** 在特定深度点查看孔径分布

#### calculate_porosity(depth_range)

**目的：** 计算指定深度范围内的总孔隙度

**原理：**
1. T2 谱的总面积与孔隙度成正比
2. 对每个深度点的 T2 谱进行积分 (求和)
3. 返回指定深度范围内的平均孔隙度

**代码示例：**
```python
def calculate_porosity(self, depth_range):
    # 筛选深度范围
    mask = (self.depth >= depth_range[0]) & (self.depth <= depth_range[1])
    spectra_subset = self.spectra[mask]
    
    # 计算每个深度点的谱面积 (孔隙度代理)
    porosity_values = np.sum(spectra_subset, axis=1)
    
    # 返回平均孔隙度
    return np.mean(porosity_values)
```

#### calculate_ffb_bfi(depth_range, t2_cutoff)

**目的：** 计算自由流体指数 (FFI) 和束缚流体指数 (BFI)

**参数：**
- `depth_range`: 深度范围
- `t2_cutoff`: T2 截止值，区分可动流体和束缚流体 (默认 100ms)

**原理：**
1. T2 < cutoff：束缚流体 (毛细管束缚)
2. T2 > cutoff：可动流体 (自由流体)
3. 分别计算两部分的谱面积占比

**返回：**
- `(FFI, BFI)`: 自由流体指数和束缚流体指数

---

## 四、在 DATA_WELL 中的集成

### 4.1 文件识别

```python
class DATA_WELL:
    # NMR 文件识别关键字
    NMR_KW = ['nmr']
    
    def scan_files(self):
        """扫描井文件夹中的 NMR 文件"""
        self.path_list_nmr = search_files_by_criteria(
            self.well_path,
            name_keywords=self.NMR_KW,
            file_extensions=['.csv'],
            all_keywords=False
        )
```

### 4.2 数据初始化

```python
def init_NMR(self, path: str = ''):
    """
    初始化 NMR 数据对象 (懒加载)
    
    参数:
        path: NMR 文件路径，为空时使用默认路径
    """
    if path not in self.NMR_dict:
        target_path = path if path else self.path_list_nmr[0]
        self.NMR_dict[target_path] = DataNMR(
            path=target_path,
            well_name=self.WELL_NAME
        )
```

### 4.3 数据访问接口

```python
def get_NMR(self, key: str = '', depth: Optional[List[float]] = None):
    """
    获取 NMR 数据
    
    参数:
        key: NMR 文件路径关键字
        depth: 深度范围限制
    
    返回:
        (depth_array, t2_times, spectra) 元组
    """
    self.init_NMR(key)
    obj = self._get_default_obj(self.NMR_dict, key)
    return obj.get_data(depth) if obj else None
```

### 4.4 文件搜索辅助方法

```python
def search_nmr_path_list(self, new_kw: List[str]) -> List[str]:
    """
    搜索特定 NMR 文件路径
    
    参数:
        new_kw: 额外关键字列表
    
    返回:
        匹配的文件路径列表
    """
    return self._search_path_list(
        base_list=self.path_list_nmr,
        new_kw=new_kw
    )
```

---

## 五、使用示例

### 5.1 基础使用

```python
from src_well_data.data_logging_well import DATA_WELL

# 创建单井对象
well = DATA_WELL(
    path_folder=r'F:\logging_workspace\云安 012-X18',
    WELL_NAME='云安 012-X18'
)

# 扫描文件
well.scan_files()

# 获取 NMR 文件列表
nmr_files = well.get_path_list_nmr()
print(f"找到 {len(nmr_files)} 个 NMR 文件")
```

### 5.2 获取 NMR 数据

```python
# 获取 NMR 数据
depth, t2_times, spectra = well.get_NMR(key=nmr_files[0])

print(f"深度范围：{depth[0]:.2f} - {depth[-1]:.2f} m")
print(f"T2 时间范围：{t2_times[0]:.2f} - {t2_times[-1]:.2f} ms")
print(f"谱数据形状：{spectra.shape}")  # (深度点数，T2 时间点数)
```

### 5.3 获取特定深度的 T2 谱

```python
# 获取 3250m 处的 T2 谱
spectrum_at_3250 = well.NMR_dict[nmr_files[0]].get_t2_spectrum_at_depth(3250.0)

if spectrum_at_3250 is not None:
    print(f"3250m 处的 T2 谱：{spectrum_at_3250}")
```

### 5.4 计算孔隙度

```python
# 计算 3200-3300m 范围内的平均孔隙度
porosity = well.NMR_dict[nmr_files[0]].calculate_porosity([3200, 3300])
print(f"平均孔隙度：{porosity:.2f}")
```

### 5.5 计算自由流体和束缚流体

```python
# 计算 FFI 和 BFI (T2 截止值 100ms)
ffi, bfi = well.NMR_dict[nmr_files[0]].calculate_ffb_bfi([3200, 3300], t2_cutoff=100.0)
print(f"自由流体指数 (FFI): {ffi:.2f}")
print(f"束缚流体指数 (BFI): {bfi:.2f}")
```

---

## 六、可视化集成

### 6.1 可视化配置

```python
config_nmr = {
    'X_LOG': [True, True],              # 是否使用对数坐标
    'NMR_TITLE': ['N1_谱', 'N2_谱'],    # 标题
    'X_LIMIT': [[0.1, 1000], [0.1, 1000]],  # X 轴范围 (T2 时间)
    'Y_scaling_factor': 12,             # Y 轴缩放因子
    'JUMP_POINT': 15,                   # 采样跳跃 (减少点数)
    'line_width': 0.8,                  # 线宽
    'fill_alpha': 0.5                   # 填充透明度
}
```

### 6.2 数据格式转换

```python
# 为可视化准备 NMR 数据字典
nmr_dict = {
    'depth': depth_array,
    't2_times': t2_times,
    'spectra': spectra_matrix,
    'title': 'NMR T2 谱'
}
```

### 6.3 渲染原理

1. **数据抽取：** 对大数据量进行抽样 (JUMP_POINT)
2. **坐标变换：** 支持对数坐标 (X_LOG=True)
3. **谱图绘制：** 使用 `fill_between()` 绘制填充曲线
4. **深度对齐：** Y 轴与测井曲线共享深度坐标
5. **多谱显示：** 支持多个 NMR 测量序列并排显示

### 6.4 可视化效果

```
        T2 谱 (对数坐标)
        ← 小孔径    大孔径 →
     0.1ms ─────────── 1000ms
     │    ╱╲
     │   ╱  ╲╱╲
3200m│──╱─────╲─╲──────  ← T2 谱曲线
     │ ╱       ╲_╲
     │╱          ╲
     └────────────────
```

---

## 七、NMR 测井基础

### 7.1 T2 弛豫时间物理意义

| T2 范围 | 孔隙类型 | 流体性质 |
|---------|----------|----------|
| < 3ms | 微孔隙 | 束缚水 |
| 3-33ms | 小孔隙 | 束缚流体 |
| 33-100ms | 中孔隙 | 可动流体 (部分) |
| 100-1000ms | 大孔隙 | 自由流体 |
| > 1000ms | 裂缝/溶洞 | 自由流体 |

### 7.2 关键参数解释

| 参数 | 含义 | 典型值 |
|------|------|--------|
| **T2 截止值** | 区分可动/束缚流体的阈值 | 33ms (砂岩), 100ms (碳酸盐岩) |
| **FFI** | 自由流体指数 (Free Fluid Index) | 0-1 |
| **BFI** | 束缚流体指数 (Bound Fluid Index) | 0-1 |
| **T2 几何均值** | 孔径分布的表征参数 | ms |
| **T2 谱面积** | 与总孔隙度成正比 | p.u. |

### 7.3 应用公式

**总孔隙度估算：**
```
φ_NMR = C × Σ(Spectrum_Amplitude)
```
其中 C 为刻度系数

**渗透率估算 (SDR 模型)：**
```
k = a × φ^b × (T2_geom)^c
```
其中 T2_geom 为 T2 几何平均值

---

## 八、性能优化

### 8.1 数据抽样

NMR 谱数据通常包含大量 T2 时间点 (64-256 个)，可视化时可通过抽样减少渲染负担：

```python
# 抽样配置
JUMP_POINT = 15  # 每 15 个点取 1 个

# 应用抽样
spectra_sampled = spectra[::JUMP_POINT, :]
t2_times_sampled = t2_times[::JUMP_POINT]
```

### 8.2 深度插值

当 NMR 数据深度与常规测井深度不一致时，可进行插值对齐：

```python
from scipy.interpolate import interp1d

def interpolate_nmr_to_logging_depth(nmr_depth, nmr_spectra, target_depth):
    """将 NMR 数据插值到测井深度"""
    # 对每个 T2 时间点进行插值
    interpolated_spectra = np.zeros((len(target_depth), nmr_spectra.shape[1]))
    
    for i in range(nmr_spectra.shape[1]):
        f = interp1d(nmr_depth, nmr_spectra[:, i], kind='linear', fill_value='extrapolate')
        interpolated_spectra[:, i] = f(target_depth)
    
    return interpolated_spectra
```

---

## 九、常见问题

### Q1: NMR 数据深度与常规测井不一致？

**解决方案：**
1. 使用插值方法对齐深度
2. 在可视化时分别显示，不强制对齐
3. 检查数据采集时的深度系统是否一致

### Q2: T2 谱出现负值？

**可能原因：**
- 数据处理过程中的基线校正问题
- 噪声影响

**解决方案：**
- 对负值进行截断处理 (`np.maximum(spectra, 0)`)
- 检查原始数据质量

### Q3: 如何确定 T2 截止值？

**方法：**
1. **岩心刻度法：** 对比 NMR 与岩心毛管压力数据
2. **经验值法：** 砂岩 33ms，碳酸盐岩 100ms
3. **T2 谱形态法：** 分析 T2 谱双峰特征

---

## 十、扩展方向

### 10.1 可能的功能扩展

1. **T2 谱分解：** 识别 T2 谱中的多个峰，对应不同孔隙系统
2. **流体识别：** 结合 T1/T2 比识别油、气、水
3. **渗透率计算：** 实现 SDR、Coates 等渗透率模型
4. **差谱分析：** 双 TW 差谱识别轻质油/气

### 10.2 与其他模块的协同

- **与常规测井协同：** NMR 孔隙度 + 密度/中子孔隙度对比
- **与表格数据协同：** NMR 参数 + 岩性标签建立预测模型
- **与 FMI 协同：** NMR 孔隙度 + FMI 裂缝识别综合解释

---

## 十一、参考资源

### 11.1 推荐文献

1. Coates, G.R., et al. (1999). "NMR Logging: Principles and Applications"
2. Dunn, K.J., et al. (2002). "Nuclear Magnetic Resonance: Petrophysical and Logging Applications"

### 11.2 常用软件

- MestReNova (NMR 数据处理)
- Techlog (测井综合解释)
- GeoFrame (斯伦贝谢测井平台)

---

*最后更新：2026-03-14*  
*Well Logging Visualization System - NMR Module v1.0*
