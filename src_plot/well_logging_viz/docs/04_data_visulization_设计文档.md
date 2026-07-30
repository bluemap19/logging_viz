# data_visulization.py 设计文档

## 1. 概述

**文件名**：`data_visulization.py`  
**核心类**：`WellLogVisualizer`  
**设计目标**：基于 Matplotlib 构建测井多面板联动可视化系统，支持测井曲线、FMI 图像、NMR 谱、岩性分类和岩心杆状图  
**新增功能**：岩心实验数据杆状图叠加绘制

---

## 2. 类设计

### 2.1 初始化

```python
def __init__(
    self,
    data_manager: LoggingDataManager = None,
    config_logging = {},
    config_fmi: Dict[str, Any] = {},
    config_nmr: Dict[str, Any] = {},
    config_type: Dict[str, Any] = {},
    config_core: Dict[str, Any] = {}    # 新增：岩心配置
) -> None:
```

### 2.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `fig` | `plt.Figure` | Matplotlib Figure 实例 |
| `axs` | `np.ndarray` | 所有子图 Axes 对象数组 |
| `config_logging/fmi/nmr/type/core` | `dict` | 各类型绘图配置 |
| `n_curve/fmi/type/nmr_panels` | `int` | 各类型面板数量 |
| `current_depth_range` | `tuple` | 当前深度窗口 |
| `logging_data_windows` | `DataFrame` | 当前窗口的常规测井数据 |
| `core_line_groups` | `List[List[Line2D]]` | **新增**：岩心曲线线组列表 |

---

## 3. 面板布局系统

### 3.1 布局计算 — `_calculate_subplot_count()`

```python
def _calculate_subplot_count(self) -> int:
    return self.n_fmi_panels + self.n_curve_panels + self.n_type_panels + self.n_nmr_panels
```

### 3.2 布局顺序

```
axs[0]              ~  axs[n_fmi_panels-1]        # FMI 电成像面板
axs[n_fmi_panels]   ~  axs[n_fmi_panels + n_curve-1]  # 常规曲线面板
axs[...]            ~  axs[... + n_type -1]        # 岩性分类面板
axs[last-n_nmr]    ~  axs[last]                   # NMR 谱面板
```

### 3.3 共享 Y 轴策略

所有面板共享深度（Y）轴，实现同步缩放和滚动：

```python
fig, axs = plt.subplots(
    1, n_subplots,
    figsize=(n_subplots * 2.0 + 1.0, 10),
    sharey=True  # 共享 Y 轴
)
```

---

## 4. 分类面板条带渲染 — `_batch_render_classification()`

### 4.0 问题背景：条带间隙 bug

**现象**：`config_type = {'types_cols': 'auto'}` 时，分类条带之间存在空白，不连续。

**根因**：原实现按分类值 `groupby` 后遍历，破坏了原始深度顺序，导致被其他值隔开的同类别点各自画矩形，条带之间产生间隙。

**修复方案**：改为按原始深度顺序扫描，将相邻且类别相同的点**合并为一条连续条带**，边界取相邻点的中点，彻底消除间隙。

### 4.0.1 修复前（错误）

```python
# 按分类值 groupby —— 破坏深度顺序
class_groups = visible_data.groupby(class_col)
for class_val, group in class_groups:
    for depth in group[depth_col]:  # 深度顺序被打乱
        y_bottom = depth - resolution / 2
        y_top = depth + resolution / 2
        # 相邻同类别点各自画矩形 → 条带间有间隙
```

### 4.0.2 修复后（正确）

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

### 4.0.3 视觉效果对比

```
修复前：  [========]     [===]  [==]   ← 有间隙
修复后：  [===============]         ← 紧密连接
```

### 4.1 渲染流程

### 4.1 首次渲染 — `visualize()`

```python
def visualize(self, top_depth: float, bottom_depth: float) -> None:
    # Step 1: 获取可见数据
    self.logging_data_windows = data_manager.get_visible_logging_data(top, bottom)
    self.current_depth_range = (top_depth, bottom_depth)

    # Step 2: 计算面板数量
    n_subplots = self._calculate_subplot_count()

    # Step 3: 创建 Figure 和 Axes
    self.fig, self.axs = plt.subplots(..., sharey=True)

    # Step 4: 绑定事件（深度拖动刷新）
    self._bind_drag_events()

    # Step 5: 绘制所有面板
    self._plot_all_fmi_panels()        # FMI
    self._plot_all_curve_panels()       # 常规曲线
    self._plot_all_classification_panels()  # 岩性分类（连续条带，无间隙）
    self._plot_all_nmr_panels()        # NMR 谱
    self._plot_all_core_lines()         # 新增：岩心杆状图

    # Step 6: 美化布局
    self._setup_axes()
```

### 4.2 刷新更新 — `_update_display()`

深度拖动时，仅更新数据，不重建 Figure：

```python
def _update_display(self, top: float, bottom: float):
    # 获取新窗口数据
    self.logging_data_windows = self.data_manager.get_visible_logging_data(top, bottom)
    self.current_depth_range = (top, bottom)

    # 更新各面板
    self._update_curve_display()        # 重绘曲线
    self._update_fmi_display()          # 重绘 FMI
    self._update_nmr_display()         # 重绘 NMR
    self._update_core_lines()          # 新增：更新岩心线
```

---

## 5. 岩心曲线道索引映射 — `_get_curve_axes_indices()`

### 5.1 设计背景

岩心杆状图需要知道叠加在哪一道（哪个 Axes）上。曲线道的 Axes 索引由 FMI 面板数量决定。

### 5.2 映射逻辑

```python
def _get_curve_axes_indices(self) -> Dict[int, int]:
    mapping = {}
    idx = self.n_fmi_panels  # 曲线道从 FMI 面板之后开始
    for i in range(len(self.config_logging['curves_plot'])):
        mapping[i] = idx  # 曲线道 i -> axs[idx]
        idx += 1
    return mapping
```

### 5.3 示例

```
n_fmi_panels = 2

mapping = {
    0: 2,  # 曲线道0 -> axs[2]
    1: 3,  # 曲线道1 -> axs[3]
    2: 4,  # 曲线道2 -> axs[4]
}
```

---

## 7. 岩心杆状图叠加绘制

> 详见 `docs/05_岩心数据叠加绘制_设计文档.md`

---

## 8. 拖动交互 — `_bind_drag_events()`

### 7.1 事件绑定

```python
def _bind_drag_events(self):
    # 缩放事件
    self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
    # 按钮按下
    self.fig.canvas.mpl_connect('button_press_event', self._on_button_press)
    # 拖动
    self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
    # 释放
    self.fig.canvas.mpl_connect('button_release_event', self._on_button_release)
```

### 7.2 深度缩放

鼠标滚轮控制深度缩放（调整上下限），触发 `_update_display()` 刷新所有面板。

---

## 9. 坐标轴美化 — `_setup_axes()`

```python
def _setup_axes(self):
    # 设置 Y 轴为深度方向（倒置，大深度在上）
    for ax in self.axs:
        ax.invert_yaxis()

    # 隐藏多余刻度标签
    # FMI 面板：隐藏 Y 刻度（仅最左面板显示）
    # 曲线面板：隐藏 Y 刻度
    # NMR 面板：隐藏 Y 刻度
    # 分类面板：显示 Y 刻度

    # 网格线
    ax.grid(True, alpha=0.3)
```

---

## 9. 完整调用示例

```python
from well_logging_viz import LoggingDataManager, WellLogVisualizer
import pandas as pd

# 构造数据
logging_data = pd.DataFrame({
    'DEPTH': range(1000, 2000),
    'GR': np.random.rand(1000) * 100,
    'RT': np.random.rand(1000) * 1000,
    'CORE_GR': np.where(np.random.rand(1000) > 0.9, np.random.rand(1000) * 150, np.nan),
    'CORE_RT': np.where(np.random.rand(1000) > 0.9, np.random.rand(1000) * 100, np.nan),
})

# 配置
config_logging = {
    'curves_plot': ['GR', 'RT'],
    'depth_col': 'DEPTH'
}

config_core = {
    'core_curves': ['CORE_GR', 'CORE_RT'],
    'plot_index_list': [0, 1],
    'thicknesses_config': [2.0, 2.5],
    'colors_config': ['#FF4500', '#1E90FF'],
    'range_config': [[0, 150], [0.5, 100]]
}

# 初始化
data_manager = LoggingDataManager(logging_data=logging_data)
viz = WellLogVisualizer(
    data_manager=data_manager,
    config_logging=config_logging,
    config_core=config_core
)

# 渲染
viz.visualize(top_depth=1000, bottom_depth=1100)
```
