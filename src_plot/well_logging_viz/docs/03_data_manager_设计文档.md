# data_manager.py 设计文档

## 1. 概述

**文件名**：`data_manager.py`  
**核心类**：`LoggingDataManager`  
**设计目标**：作为测井可视化系统的**数据中枢**，管理原始数据输入、验证、缓存访问和配置智能检查与自动补全  
**新增功能**：岩心实验数据配置验证与自适应补全

---

## 2. 类设计

### 2.1 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `logging_data` | `pd.DataFrame` | 常规测井数据（含深度列、曲线列、岩心列） |
| `fmi_data` | `dict` | FMI 电成像数据：`{'depth': np.ndarray, 'image_data': [np.ndarray]}` |
| `nmr_data` | `dict` | NMR 谱数据：`{'depth': np.ndarray, 'nmr_data': [np.ndarray]}` |
| `config_core` | `dict` | **新增**：岩心实验数据配置 |
| `cache_system` | `EnhancedWellLogCache` | LRU 缓存系统 |

### 2.2 初始化流程

```
__init__(logging_data, fmi_data, nmr_data, config_core)
    │
    ├─ _validate_logging_data()      验证常规测井 DataFrame
    ├─ _validate_fmi_data()           验证 FMI 数据结构
    ├─ _validate_nmr_data()           验证 NMR 数据结构
    ├─ config_core = {}              初始化空配置
    └─ _get_depth_limits()            计算全井深度范围
```

### 2.3 岩心配置初始化

```python
def __init__(self, ..., config_core: dict = None):
    # ...
    self.config_core = config_core if config_core is not None else {}
```

岩心配置在 `plot_config_check()` 中统一验证和补全。

---

## 3. 核心方法

### 3.1 可见数据提取

| 方法 | 说明 |
|------|------|
| `get_visible_logging_data(top, bottom)` | 获取常规测井可见窗口（带缓存） |
| `get_visible_fmi_data(top, bottom, max_vertical_points)` | 获取 FMI 图像可见窗口（带缓存） |
| `get_visible_nmr_data(top, bottom, spectral_density)` | 获取 NMR 谱可见窗口（带缓存） |

**缓存查找流程**：

```
请求深度范围 (top, bottom)
    ↓
cache_system.get_xxx_data((top, bottom))
    │
    ├─ 命中 → 直接返回缓存数据
    └─ 未命中 → 从原始数据提取切片 → 压缩 → 存入缓存 → 返回
```

**深度切片压缩**：当可见数据点数超过目标点数时，通过 `np.linspace` 等间距采样：

```python
def _compress_depths(self, depths, target_points):
    if len(depths) <= target_points:
        return depths
    indices = np.linspace(0, len(depths)-1, target_points).astype(int)
    return depths[indices]
```

### 3.2 配置检查系统 — `plot_config_check()`

**设计原则**：配置即契约——用户只需指定必要参数，其余由系统智能推断。

#### 3.2.1 签名

```python
def plot_config_check(
    self,
    config_logging: Dict = None,
    config_fmi: Dict = None,
    config_nmr: Dict = None,
    config_type: Dict = None,
    config_core: Dict = None    # 新增
) -> Dict[str, Dict]
```

#### 3.2.2 处理流程

```
┌─────────────────────────────────────────────────────┐
│                   plot_config_check()                 │
├─────────────────────────────────────────────────────┤
│ 1. _get_default_plot_configs()                       │
│    生成 5 个默认配置模板                              │
│    (logging / fmi / nmr / type / core)               │
│                                                     │
│ 2. _deep_merge_configs(default, user)                │
│    递归合并用户配置与默认模板                         │
│                                                     │
│ 3. _validate_and_adapt_xxx_config(merged)            │
│    ├─ _validate_and_adapt_logging_config()           │
│    ├─ _validate_and_adapt_fmi_config()               │
│    ├─ _validate_and_adapt_nmr_config()                │
│    ├─ _validate_and_adapt_type_config()              │
│    └─ _validate_and_adapt_core_config()    ← 新增    │
│                                                     │
│ 4. _log_config_check_results()                       │
│    打印配置检查报告                                   │
└─────────────────────────────────────────────────────┘
```

#### 3.2.3 默认配置模板 — `core` 配置

```python
'core': {
    'plot_index_list': [],      # 岩心叠加在哪几道
    'core_curves': [],          # 岩心曲线列名
    'thicknesses_config': [],   # 杆粗细（线宽）
    'colors_config': [],       # 杆颜色
    'axis_config': [],          # 坐标轴是否 log
    'range_config': []          # 范围 [min, max]
}
```

---

## 4. 岩心配置验证 — `_validate_and_adapt_core_config()`

### 4.1 设计背景

岩心实验数据特点：

- **稀疏性**：大部分深度点为 NaN，仅在取心深度有数据
- **共存储**：与常规曲线共存于 `logging_data` DataFrame 的特定列
- **叠加绘制**：不占用独立面板，叠加在已有曲线道的 X 轴右侧

### 4.2 验证逻辑（8 步）

```
Step 1: 空配置检查
        ↓ curves_plot 为空 → 静默跳过
Step 2: 列存在性验证
        ↓ core_curves 中存在的列 → valid_core_curves
Step 3: plot_index_list 智能分配
        ↓ 未指定时，按数量均分到各常规曲线道
Step 4: 粗细配置补全
        ↓ 不足数量时，填充默认值 2.0
Step 5: 颜色配置补全
        ↓ 不足数量时，填充默认颜色序列
        ['#1E90FF', '#FF6347', '#32CD32', '#FFD700', '#9370DB']
Step 6: 透明度配置补全（alphas_config）
        ↓ 不足数量时，填充默认值 1.0（完全不透明）
Step 7: 范围配置自动推断
        ↓ 从实际数据 min/max 推断，添加 5% 边距
Step 8: 坐标轴配置补全（axis_config）
        ↓ 不足数量时，填充 False（不使用对数轴）
```

### 4.3 返回值

验证后的完整 `config_core` 字典，确保所有列表长度一致。

---

## 5. `cal_plot_num()` — 面板数量计算

### 5.1 设计背景

可视化层需要提前知道需要创建多少个子图面板，以构建正确的 Matplotlib Figure 布局。

### 5.2 逻辑

```python
def cal_plot_num(self):
    # 常规曲线面板
    self.n_curve_panels = len(config_logging['curves_plot'])  # 0 ~ N

    # 分类面板
    self.n_type_panels = len(config_type['types_cols'])       # 0 ~ 2

    # FMI 面板
    self.n_fmi_panels = len(fmi_data['image_data'])            # 0 ~ 5

    # NMR 面板
    self.n_nmr_panels = len(nmr_data['nmr_data'])               # 0 ~ 4

    # 岩心：无独立面板（叠加绘制）
    # self.n_core_panels = 0

    return n_curve, n_type, n_fmi, n_nmr
```

### 5.3 典型返回值

| 场景 | 返回值 |
|------|--------|
| 仅常规曲线 | `(8, 0, 0, 0)` |
| 全功能 | `(8, 2, 5, 4)` |

---

## 6. 配置示例

```python
# 岩心配置示例
config_core = {
    'plot_index_list': [0, 1, 2],       # 叠加在第0、1、2道
    'core_curves': ['CORE_GR', 'CORE_RT'],  # 两列岩心数据
    'thicknesses_config': [2.5, 3.0],    # 杆粗细
    'colors_config': ['#FF4500', '#1E90FF'],  # 橙红色、蓝色
    'axis_config': [False, False],      # 不使用对数轴
    'range_config': [[0, 150], [0.5, 100]]  # GR范围、RT范围
}
```
