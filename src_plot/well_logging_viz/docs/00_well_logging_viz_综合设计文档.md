# well_logging_viz 模块 — 综合设计文档

> **版本**：v1.0  
> **维护者**：Cuka (Agent for Dr. Fuhao Zhang)  
> **最后更新**：2026-07-30  
> **适用范围**：`D:\GitHub\logging_viz\src_plot\well_logging_viz\`

---

## 目录

1. [模块概览](#1-模块概览)
2. [文件清单与职责](#2-文件清单与职责)
3. [模块间交互关系](#3-模块间交互关系)
4. [cache_logging.py — 缓存系统](#4-cache_loggingpy--缓存系统)
5. [data_manager.py — 数据管理器](#5-data_managerpy--数据管理器)
6. [data_visulization.py — 可视化器](#6-data_visulizationpy--可视化器)
7. [岩心数据叠加绘制设计](#7-岩心数据叠加绘制设计)
8. [岩心配置格式详解](#8-岩心配置格式详解)
9. [已知问题与修复记录](#9-已知问题与修复记录)
10. [性能考量](#10-性能考量)
11. [扩展方向](#11-扩展方向)

---

## 1. 模块概览

`well_logging_viz` 是 `logging_viz` 系统的**核心可视化模块**，负责将常规测井曲线、FMI 电成像图像、NMR 核磁共振谱、岩性分类数据以及岩心实验数据渲染为专业级交互式测井图。

### 1.1 设计目标

| 目标 | 说明 |
|------|------|
| **多数据融合** | 常规曲线 + 成像 + 核磁 + 分类 + 岩心，一图尽览 |
| **交互式导航** | 鼠标滚轮缩放深度窗口，滑动条调整窗口大小 |
| **配置即契约** | 用户只需指定必要参数，其余系统智能推断 |
| **性能优先** | LRU 缓存 + zlib 压缩 + 对象复用，避免重复渲染 |
| **零依赖扩展** | 岩心数据叠加绘制，不增加面板，不破坏架构 |

### 1.2 架构层次

```
┌──────────────────────────────────────────────────────┐
│              上层调用 / TEMP_TEST_VIZ.py             │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│                 LoggingDataManager                    │
│              data_manager.py                          │
│  ┌──────────────────────────────────────────────┐   │
│  │         EnhancedWellLogCache                  │   │
│  │             cache_logging.py                  │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│                  WellLogVisualizer                    │
│              data_visulization.py                     │
└──────────────────────────────────────────────────────┘
```

---

## 2. 文件清单与职责

| 文件 | 规模 | 核心类 | 职责 |
|------|------|--------|------|
| `__init__.py` | 0 字节 | — | Python 包标记文件（空） |
| `cache_logging.py` | ~14.8 KB | `EnhancedWellLogCache` | 三层 LRU 缓存（常规/FMI/NMR），zlib 压缩，命中率统计 |
| `data_manager.py` | ~61.6 KB | `LoggingDataManager` | 数据验证、配置检查、可见窗口提取、缓存调度 |
| `data_visulization.py` | ~70.3 KB | `WellLogVisualizer` | Matplotlib 多面板绑定、交互事件、数据渲染 |
| `TEMP_TEST_VIZ.py` | ~6.7 KB | — | 集成测试脚本，构造真实/模拟数据并调用完整流程 |

---

## 3. 模块间交互关系

### 3.1 数据流

```
原始数据（DataFrame / np.ndarray）
    │
    │ 传入
    ▼
LoggingDataManager.__init__()
    ├─ _validate_logging_data()     验证 DataFrame 结构
    ├─ _validate_fmi_data()          验证图像数据格式
    ├─ _validate_nmr_data()          验证 NMR 谱数据格式
    └─ _get_depth_limits()           计算全井深度范围
    │
    │ 初始化
    ▼
EnhancedWellLogCache (内置)
    ├─ _data_cache:  OrderedDict → DataFrame（无压缩）
    ├─ _fmi_cache:   OrderedDict → bytes（zlib 压缩）
    └─ _nmr_cache:   OrderedDict → bytes（zlib 压缩）
    │
    │ 上层调用
    ▼
WellLogVisualizer.visualize()
    │
    ├─ data_manager.get_visible_logging_data()  → 带缓存的 DataFrame
    ├─ data_manager.get_visible_fmi_data()       → 带缓存的 List[np.ndarray]
    ├─ data_manager.get_visible_nmr_data()       → 带缓存的 List[np.ndarray]
    ├─ data_manager.plot_config_check()          → 配置验证 + 补全
    │
    │ 渲染
    ▼
Matplotlib Figure（多面板绑定图）
```

### 3.2 配置流

```
用户传入部分配置
    │
    ▼
LoggingDataManager.plot_config_check()
    ├─ _get_default_plot_configs()     生成默认模板
    ├─ _deep_merge_configs()           深度合并（用户优先）
    ├─ _validate_and_adapt_xxx_config() 各类型独立验证
    └─ _log_config_check_results()     日志记录
    │
    ▼
WellLogVisualizer.__init__()
    ├─ 接收完整配置
    ├─ 计算面板数量
    └─ 初始化渲染状态
```

### 3.3 岩心数据特殊流程

```
config_core 配置字典
    │
    ▼
_plot_all_core_lines()     初始绘制（visualize() 中调用）
_update_core_lines()       深度刷新（_update_display() 中调用）
    │
    ├─ 不走独立面板（无 n_core_panels）
    ├─ 叠加在曲线道的 X 轴最左侧
    └─ 复用 Line2D 对象（set_data 更新位置）
```

---

## 4. cache_logging.py — 缓存系统

### 4.1 类设计

**`CacheConfig`** — 缓存配置数据类

```python
@dataclass
class CacheConfig:
    enabled: bool = True              # 缓存总开关
    max_size: int = 100               # 常规数据缓存最大条目数
    fmi_max_size: int = 50            # FMI 缓存最大条目数（数据量大，限制更严）
    nmr_max_size: int = 100           # NMR 缓存最大条目数
    compression_level: int = 1        # zlib 压缩级别（1-9，1 最快）
```

**`EnhancedWellLogCache`** — 缓存核心类

| 属性 | 类型 | 说明 |
|------|------|------|
| `_data_cache` | `OrderedDict` | 常规测井 DataFrame 缓存（无压缩） |
| `_fmi_cache` | `OrderedDict` | FMI 图像缓存（zlib 压缩存储） |
| `_nmr_cache` | `OrderedDict` | NMR 谱缓存（zlib 压缩存储） |
| `_core_stats` | `dict` | 岩心访问统计（复用 DataFrame 缓存） |
| `stats` | `dict` | 全局命中率统计 |

### 4.2 三层缓存架构

```
┌─────────────────────────────────────────────────────────┐
│              EnhancedWellLogCache                         │
├───────────────┬──────────────────┬──────────────────────┤
│  _data_cache  │   _fmi_cache     │   _nmr_cache         │
│  OrderedDict  │   OrderedDict    │   OrderedDict        │
│               │                  │                      │
│  key: tuple   │   key: tuple     │   key: tuple         │
│  value:       │   value: bytes   │   value: bytes       │
│    DataFrame  │    (zlib压缩)    │    (zlib压缩)        │
│               │                  │                      │
│  无压缩       │   zlib 压缩     │   zlib 压缩          │
│  LRU 淘汰    │   LRU 淘汰      │   LRU 淘汰           │
└───────────────┴──────────────────┴──────────────────────┘
```

### 4.3 LRU 淘汰策略

当缓存条目数达到 `max_size` 时，自动淘汰最旧条目：

```python
while len(self._data_cache) > self.config.max_size:
    removed_key, _ = self._data_cache.popitem(last=False)  # 淘汰最旧
    self._data_cache.move_to_end(key)  # 命中后移到最后（最新）
```

### 4.4 zlib 压缩机制

**压缩流程**：

```
List[np.ndarray]（FMI/NMR 原始数据）
    │
    ├─ pickle.dumps()    → 原始字节流
    ├─ zlib.compress(level=1)  → 压缩字节流
    ├─ 存储到 OrderedDict
    │
    ├─ zlib.decompress()  → 压缩字节流
    └─ pickle.loads()     → 还原 List[np.ndarray]
```

**压缩比参考**：

| 数据类型 | 典型压缩比 | 说明 |
|----------|-----------|------|
| FMI 电阻率图像 | 3-8x | 图像数据冗余度高 |
| NMR T2 谱 | 2-4x | 浮点数组冗余 |
| 常规测井曲线 | 1.5-3x | 不推荐压缩（CPU > IO 收益） |

### 4.5 岩心数据缓存策略

**设计决策**：岩心数据不单独存储，复用常规测井 DataFrame 缓存。

**理由**：

- 岩心列与常规曲线列共存于同一 DataFrame，单独缓存增加管理复杂度
- 岩心访问通过 `get_visible_logging_data()` 自然命中 DataFrame 缓存
- 通过 `record_core_access()` 单独统计岩心数据访问次数

### 4.6 核心方法一览

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_logging_data` | `(depth_range) → DataFrame\|None` | 获取常规测井缓存 |
| `set_logging_data` | `(depth_range, data) → None` | 设置常规测井缓存 |
| `get_fmi_data` | `(depth_range) → List[np.ndarray]\|None` | 获取 FMI 缓存（含解压） |
| `set_fmi_data` | `(depth_range, fmi_data) → None` | 设置 FMI 缓存（含压缩） |
| `get_nmr_data` | `(depth_range) → List[np.ndarray]\|None` | 获取 NMR 缓存（含解压） |
| `set_nmr_data` | `(depth_range, nmr_data) → None` | 设置 NMR 缓存（含压缩） |
| `record_core_access` | `() → None` | 记录岩心数据访问 |
| `get_cache_stats` | `() → dict` | 命中率、压缩比、内存节省 |
| `get_memory_usage` | `() → dict` | 各缓存内存占用（MB） |
| `clear_cache` | `() → None` | 清空所有缓存 |

---

## 5. data_manager.py — 数据管理器

### 5.1 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `logging_data` | `pd.DataFrame` | 常规测井数据（含深度列、曲线列、岩心列） |
| `fmi_data` | `dict` | FMI 电成像：`{'depth': np.ndarray, 'image_data': [np.ndarray]}` |
| `nmr_data` | `dict` | NMR 谱：`{'depth': np.ndarray, 'nmr_data': [np.ndarray]}` |
| `config_core` | `dict` | 岩心实验数据配置 |
| `cache_system` | `EnhancedWellLogCache` | LRU 缓存系统 |
| `depth_min / depth_max` | `float` | 全井深度范围 |

### 5.2 性能配置

```python
PERFORMANCE_CONFIG = {
    'cache_enabled': True,
    'max_cache_size': 500,       # 常规数据缓存最大条目
    'compression_level': 1,      # zlib 压缩级别
    'fmi_cache_size': 50,        # FMI 缓存最大条目
    'nmr_cache_size': 100,       # NMR 缓存最大条目
}
```

### 5.3 配置检查系统 — `plot_config_check()`

**设计原则**：配置即契约——用户只需指定必要参数，其余由系统智能推断。

**处理流程**：

```
用户传入部分配置
    │
    ▼
_get_default_plot_configs()     生成 5 个默认配置模板
    │
    ▼
_deep_merge_configs(default, user)  深度合并（递归，用户优先）
    │
    ▼
_validate_and_adapt_xxx_config()   各类型独立验证 + 智能补全
    ├─ _validate_and_adapt_logging_config()
    ├─ _validate_and_adapt_fmi_config()
    ├─ _validate_and_adapt_nmr_config()
    ├─ _validate_and_adapt_type_config()
    └─ _validate_and_adapt_core_config()  ← 岩心配置验证
    │
    ▼
_log_config_check_results()     打印配置检查报告
    │
    ▼
返回完整 validated_configs
```

### 5.4 岩心配置验证 — `_validate_and_adapt_core_config()`

**验证逻辑（8 步）**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 空配置检查 | `core_curves` 为空 → 静默跳过 |
| 2 | 列存在性验证 | 仅保留 `logging_data` 中存在的列 |
| 3 | `plot_index_list` 智能分配 | 未指定时，按数量均分到各常规曲线道 |
| 4 | `thicknesses_config` 补全 | 不足时填充默认值 `2.0` |
| 5 | `colors_config` 补全 | 不足时填充默认颜色序列 |
| 6 | `alphas_config` 补全 | 不足时填充默认值 `1.0`（完全不透明） |
| 7 | `range_config` 自动推断 | 从数据 `min/max` 推断，添加 5% 边距 |
| 8 | `axis_config` 补全 | 不足时填充 `False`（不使用对数轴） |

**默认岩心颜色序列**：

```python
['#1E90FF', '#FF6347', '#32CD32', '#FFD700', '#9370DB']
#  道奇蓝  番茄红   酸橙绿   金色     亮紫色
```

### 5.5 可见数据提取

**缓存查找流程**：

```
请求深度范围 (top, bottom)
    │
    ├─ cache_system.get_xxx_data((top, bottom))
    │   ├─ 命中 → 直接返回（LRU 更新）
    │   └─ 未命中 → 继续
    │
    ▼
从原始数据提取切片（depth_mask 过滤）
    │
    ├─ 数据量大？→ _compress_xxx_vertically() 垂直压缩
    │
    ▼
cache_system.set_xxx_data()  存入缓存
    │
    ▼
返回可见数据
```

**深度切片压缩**：当可见数据点数超过目标点数时，通过 `np.linspace` 等间距采样：

```python
def _compress_depths(self, depths, target_points):
    if len(depths) <= target_points:
        return depths
    indices = np.linspace(0, len(depths)-1, target_points).astype(int)
    return depths[indices]
```

### 5.6 `cal_plot_num()` — 面板数量计算

```python
def cal_plot_num(self):
    n_curve = len(config_logging['curves_plot'])    # 常规曲线面板数
    n_type  = len(config_type['types_cols'])        # 分类面板数
    n_fmi   = len(fmi_data['image_data'])            # FMI 面板数
    n_nmr   = len(nmr_data['nmr_data'])              # NMR 面板数
    # 岩心：无独立面板（叠加绘制）
    # n_core = 0
    return n_curve, n_type, n_fmi, n_nmr
```

---

## 6. data_visulization.py — 可视化器

### 6.1 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `fig` | `plt.Figure` | Matplotlib Figure 实例 |
| `axs` | `np.ndarray` | 所有子图 Axes 对象数组 |
| `logging_data_windows` | `DataFrame` | 当前深度窗口的测井数据 |
| `core_line_groups` | `List[List[Line2D]]` | 岩心曲线线组列表（每个曲线一组） |
| `depth_position` | `float` | 当前深度窗口起始位置 |
| `window_size` | `float` | 当前深度窗口大小 |

### 6.2 面板布局系统

**布局计算**：

```python
def _calculate_subplot_count(self) -> int:
    return self.n_fmi_panels + self.n_curve_panels + self.n_type_panels + self.n_nmr_panels
    # 岩心不占独立面板
```

**布局顺序**：

```
axs[0]              ~ axs[n_fmi-1]              # FMI 电成像面板
axs[n_fmi]          ~ axs[n_fmi+n_curve-1]      # 常规曲线面板
axs[n_fmi+n_curve]  ~ axs[...+n_type-1]         # 岩性分类面板
axs[last-n_nmr+1]   ~ axs[last]                 # NMR 谱面板
```

**共享 Y 轴**：所有面板通过 `sharey=True` 共享深度轴，实现同步缩放和滚动。

### 6.3 渲染流程

**首次渲染** — `visualize()`：

```
Step 1: 获取可见数据
        get_visible_logging_data / fmi_data / nmr_data

Step 2: 计算面板数量
        _calculate_subplot_count()

Step 3: 创建 Figure 和 Axes
        plt.subplots(..., sharey=True)

Step 4: 优化 FMI 渲染
        _optimize_fmi_rendering()

Step 5: 绘制所有面板
        _plot_all_fmi_panels()
        _plot_all_curves()
        _plot_all_classification_panels()
        _plot_all_nmr_panels()
        _plot_all_core_lines()       ← 岩心杆状图

Step 6: 绑定交互事件
        滑动条回调 / 滚轮缩放 / 拖动刷新

Step 7: 初始显示
        _update_display()
        plt.show()
```

**深度刷新** — `_update_display()`：

```
Step 1: 获取新窗口数据
        get_visible_logging_data(top, bottom)

Step 2: 更新各面板
        _update_curve_display()
        _update_fmi_display()
        _update_nmr_display()
        _update_core_lines()         ← 岩心杆状图刷新

Step 3: 更新深度指示器
        depth_indicator.set_text()
```

### 6.4 岩心曲线道索引映射 — `_get_curve_axes_indices()`

岩心杆状图需要知道叠加在哪一道（哪个 Axes）上。曲线道的 Axes 索引由 FMI 面板数量决定：

```python
def _get_curve_axes_indices(self) -> Dict[int, int]:
    mapping = {}
    idx = self.n_fmi_panels  # 曲线道从 FMI 面板之后开始
    for i in range(len(self.config_logging['curves_plot'])):
        mapping[i] = idx
        idx += 1
    return mapping
```

**示例**（`n_fmi_panels = 2`）：

```
mapping = {0: 2, 1: 3, 2: 4, ...}
 曲线道0 → axs[2]
 曲线道1 → axs[3]
 曲线道2 → axs[4]
```

### 6.5 分类面板条带渲染 — `_batch_render_classification()`

**问题背景**：`config_type = {'types_cols': 'auto'}` 时，分类条带之间存在空白，不连续。

**根因**：原实现按分类值 `groupby` 后遍历，破坏了原始深度顺序，导致被其他值隔开的同类别点各自画矩形，条带之间产生间隙。

**修复方案**：按原始深度顺序扫描，将相邻且类别相同的点**合并为一条连续条带**，边界取相邻点的中点。

**修复前（错误）**：

```python
# 按分类值 groupby —— 破坏深度顺序
class_groups = visible_data.groupby(class_col)
for class_val, group in class_groups:
    for depth in group[depth_col]:  # 深度顺序被打乱
        y_bottom = depth - resolution / 2
        y_top = depth + resolution / 2
        # 相邻同类别点各自画矩形 → 条带间有间隙
```

**修复后（正确）**：

```python
# 按原始深度顺序扫描
i = 0
while i < n:
    if pd.isna(class_values[i]):
        i += 1
        continue

    class_int = int(class_values[i])
    strip_start = depths[i]

    # 向后扫描所有连续同类别的点
    j = i + 1
    while j < n and int(class_values[j]) == class_int:
        j += 1

    strip_end = depths[j - 1]

    # 边界取相邻点中点（消除间隙）
    y_bottom = (strip_start + depths[i-1]) / 2 if i > 0 else strip_start - res/2
    y_top    = (strip_end   + depths[j])   / 2 if j < n else strip_end   + res/2

    # 画一条合并后的连续矩形
    vertices = [[0, y_bottom], [xmax, y_bottom], [xmax, y_top], [0, y_top]]
    i = j  # 跳到下一段
```

**视觉效果对比**：

```
修复前：  [========]     [===]  [==]   ← 有间隙
修复后：  [===============]         ← 紧密连接
```

---

## 7. 岩心数据叠加绘制设计

> 详见 `docs/05_岩心数据叠加绘制_设计文档.md`

### 7.1 叠加位置策略

岩心杆状图从曲线道的**最左侧（X 轴最小值）**开始绘制，向右延伸：

```
X 轴布局：
[0%─────────────────────────────────────────90%────100%]
|          岩心杆状图（从左侧起始）           | 空白边距 |
```

- 岩心杆从道最左侧 `xlim[0]` 开始
- 杆最大宽度为道宽的 **90%**
- 剩余 10% 作为右侧空白边距

### 7.2 杆粗细映射

杆的粗细（水平短线长度）由岩心数据值归一化决定：

```
归一化值 = (value - min) / (max - min)
杆宽度   = X轴范围 × 90% × 归一化值
```

即：**值越大 → 杆越长（粗），值越小 → 杆越短（细）**

### 7.3 Z 轴层级

| 元素 | zorder | 说明 |
|------|--------|------|
| 背景 | 0 | Axes 背景 |
| 网格 | 1 | 网格线 |
| 分类条带 | 3 | 岩性分类填充 |
| FMI 图像 | 4 | 电成像图像 |
| 常规曲线 | 5 | 测井曲线 |
| **岩心杆** | **10** | 岩心短线（最上层） |

### 7.4 初始绘制 — `_plot_all_core_lines()`

**流程**：

```
_plot_all_core_lines()
    │
    ├─ 获取曲线道轴索引映射
    │   _get_curve_axes_indices()
    │
    ├─ 遍历每个岩心曲线配置
    │   │
    │   ├─ 获取目标 Axes
    │   ├─ 读取深度 + 值数据
    │   ├─ 过滤 NaN 和异常值
    │   ├─ 归一化计算杆长度
    │   └─ plt.plot() 绘制水平短线
    │
    └─ 保存所有 Line2D 对象到 core_line_groups
```

### 7.5 刷新更新 — `_update_core_lines()`

**策略：复用已有 Line2D 对象**

```
旧线组（n_existing 条）
    │
    ├─ 逐点遍历：
    │   if 有旧线 → set_data() 更新位置 + set_visible(True)
    │   else      → plt.plot() 新建 + append
    │
    ├─ 多余旧线 → set_visible(False) 隐藏
    │
    └─ 同步更新 core_line_groups
```

**为什么需要更新**：

- 深度窗口变化 → 新的可见深度范围 → 重新过滤数据点
- X 轴范围可能变化 → 归一化映射目标区间也变化

---

## 8. 岩心配置格式详解

### 8.1 完整配置项

```python
config_core = {
    # ========== 核心配置（必填）==========
    'core_curves': List[str],        # 岩心数据列名（存在于 logging_data）

    # ========== 绘制位置配置 ==========
    'plot_index_list': List[int],   # 叠加绘制在哪几道（曲线道索引，从0开始）
                                      # 长度应与 core_curves 一致
                                      # 未指定时自动均分到各常规曲线道

    # ========== 样式配置 ==========
    'thicknesses_config': List[float],  # 杆粗细（Line2D 线宽），默认 2.0
    'colors_config': List[str],         # 杆颜色（CSS 颜色或 Hex），默认蓝橙序列
    'alphas_config': List[float],       # 岩心杆透明度（0.0~1.0），默认 1.0（完全不透明）

    # ========== 坐标轴配置 ==========
    'axis_config': List[bool],          # 坐标轴是否 log 刻度（当前版本未使用）

    # ========== 数据范围配置 ==========
    'range_config': List[List[float]],  # [min, max] 范围，用于归一化
                                        # 未指定时从数据实际 min/max 推断（加 5% 边距）
}
```

### 8.2 配置示例

**简单场景**（仅配置岩心列，其余自动推断）：

```python
config_core = {
    'core_curves': ['CORE_GR', 'CORE_RT', 'CORE_AC']
}
```

**完整场景**（显式指定所有参数）：

```python
config_core = {
    'core_curves': ['CORE_GR', 'CORE_RT'],
    'plot_index_list': [0, 1],           # 第0道画 CORE_GR，第1道画 CORE_RT
    'thicknesses_config': [2.0, 3.0],    # 第一道杆细一些
    'colors_config': ['#FF6347', '#4169E1'],  # 番茄红、皇家蓝
    'alphas_config': [0.85, 0.70],       # 透明度设置
    'axis_config': [False, False],
    'range_config': [
        [0, 150],    # CORE_GR 范围
        [0.2, 100]   # CORE_RT 范围
    ]
}
```

**多曲线叠加到同一道**：

```python
config_core = {
    'core_curves': ['CORE_GR', 'CORE_RT', 'CORE_DEN'],
    'plot_index_list': [0, 0, 0],         # 都叠加在第0道
    'thicknesses_config': [2.0, 2.0, 1.5],
    'colors_config': ['#FF6347', '#4169E1', '#32CD32'],
    'alphas_config': [0.8, 0.8, 0.9],
    'range_config': [[0, 150], [0.2, 100], [2.0, 3.0]]
}
```

### 8.3 alpha 透明度参考

| alpha 值 | 视觉效果 |
|----------|----------|
| 1.0 | 完全不透明（默认值） |
| 0.8 | 略微透明，叠加在曲线上仍清晰可见 |
| 0.5 | 半透明，可透过看到下方曲线 |
| 0.3 | 明显透明，背景曲线清晰可见 |

---

## 9. 已知问题与修复记录

### 9.1 分类条带间隙 bug（已修复）

**问题**：`config_type = {'types_cols': 'auto'}` 时，分类条带之间存在空白，不连续。  
**根因**：按分类值 `groupby` 后遍历，破坏了原始深度顺序。  
**修复**：改为按原始深度顺序扫描，相邻同类别的点合并为连续条带，边界取相邻点中点。  
**文件**：`data_visulization.py` — `_batch_render_classification()`

### 9.2 岩心杆起始位置（已修复）

**问题**：岩心杆从 X 轴 60% 处开始，浪费了左侧空间。  
**修复**：改为从道最左侧 `xlim[0]` 开始，杆宽度映射区间从 35% 调整为 90%。  
**文件**：`data_visulization.py` — `_plot_all_core_lines()` 和 `_update_core_lines()`

### 9.3 重复代码块（✅ 已修复）

**位置**：`data_manager.py` 原第 668-675 行  
**问题**：`colors_config` 补全代码块出现两次（Step 4 和 Step 6 的注释区域内各有一份），且第二份被错误地嵌套在 `range_config` 的 `if` 块内。  
**修复**：删除错误嵌套的重复 `colors_config` 块（2026-07-30 完成）。  
**修复后状态**：✅ Step 4/5/6/7/8 均各司其职，无重复。

```python
# 重复区域（data_manager.py ~line 668-675）—— 应删除
if len(range_config) < n_curves:
    for curve_col in config['core_curves'][len(range_config):]:
        # ... colors_config 重复补全代码 ...
        config['colors_config'] = colors_config  # ← 重复！已在 Step 5 处理
```

### 9.4 `alphas_config` 验证逻辑（待补充）

**位置**：`data_manager.py` — `_validate_and_adapt_core_config()`  
**问题**：Step 6 的 `alphas_config` 补全代码在文件实际实现中存在，但设计文档需确认与代码一致。  
**状态**：代码中 Step 6 存在，但 `_deep_merge_configs` 合并时若用户已指定空列表 `[]`，智能补全逻辑可能不触发。

---

## 10. 性能考量

| 指标 | 数值 | 说明 |
|------|------|------|
| LRU 缓存查找 | O(1) | `OrderedDict` 实现 |
| FMI/NMR 压缩 | 3-8x | zlib 压缩比 |
| 岩心短线绘制 | ~2ms / 1000 点 | Matplotlib 高效 |
| 深度刷新 | ~50ms / 全图 | 复用 Line2D 对象 |
| 内存占用（缓存） | 可配置（默认 500 条） | `max_cache_size` 控制 |

**岩心数据最坏情况**：5000 个取心点 × 3 个岩心曲线 = 15000 条短线，~30ms 绘制时间，可接受。

---

## 11. 扩展方向

| 扩展 | 说明 | 优先级 |
|------|------|--------|
| **对数轴支持** | `axis_config` 对数刻度下的归一化映射 | 中 |
| **标记符号** | 不同岩性用不同标记（圆形、三角形、星形） | 低 |
| **Tooltip** | 鼠标悬停显示岩心数据值和深度 | 低 |
| **标注文字** | 在杆旁边标注数值 | 低 |
| **误差棒** | 支持上下限范围的岩心数据 | 低 |
| **线程安全** | 缓存系统加锁，支持高并发 | 中 |

---

## 附录：文件大小对照表

| 文件 | 大小 | 最后修改 |
|------|------|----------|
| `cache_logging.py` | ~14.8 KB | 2026-07-30 17:50 |
| `data_manager.py` | ~61.6 KB | 2026-07-30 19:03 |
| `data_visulization.py` | ~70.3 KB | 2026-07-30 19:15 |
| `TEMP_TEST_VIZ.py` | ~6.7 KB | 2026-07-30 19:15 |
| `__init__.py` | 0 字节 | 2026-06-19 |

---

_文档版本：v1.0 | 最后更新：2026-07-30 | Agent：Cuka_
