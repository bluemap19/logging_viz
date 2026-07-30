# `data_logging_core.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_core.py`  
**核心类：** `DataCore`  
**功能定位：** 岩心实验数据（矿物/岩性组分百分比）读取与预处理，支持 CSV/Excel/TXT 多格式。  
**代码规模：** ~340 行（不含测试代码）  
**最后更新：** 2026-07-30

---

## 1. 类架构

### 1.1 异常类与枚举

```python
class CoreException(Exception)
class FileFormat(Enum):
    CSV = '.csv'
    EXCEL = '.xlsx'
    TEXT = '.txt'
    UNKNOWN = 'unknown'
```

### 1.2 核心类：`DataCore`

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_data` | `pd.DataFrame` | 原始岩心数据 |
| `_curve_names` | `List[str]` | 列名列表（含 DEPTH 列） |
| `_file_path` | `str` | 数据文件路径 |
| `_well_name` | `str` | 井名标识 |
| `_logger` | `logging.Logger` | 日志记录器 |
| `_is_data_loaded` | `bool` | 数据加载状态 |

> ⚠️ 与早期版本对比：**`_resolution` 属性已移除**，不再计算和存储分辨率。

---

## 2. 岩心数据特点

与常规测井数据的核心区别：

| 特性 | 常规测井 | 岩心数据 |
|------|----------|----------|
| 采样密度 | 高（~0.0025m/点） | 低（~1m/点，稀疏） |
| 数据覆盖 | 连续曲线 | 离散的稀疏点 |
| 深度分布 | 规则等间隔 | 不规则间隔 |
| 分辨率 | 有（连续曲线需标注） | 无（稀疏点数据不需要） |

**典型数据示例（姬 119H2 井）：**

```
DEPTH     石英    钾长石    斜长石    黄铁矿    黏土矿物
2726.53   28.9    5.6      5.3      31.6      28.7
2727.27   61.7    7.8      2.5       8.5      13.0
2727.92   37.8    4.5      5.1      18.0      30.2
...（深度间隔约 1m，共 34 个样本）
2760.76   21.1    3.7      19.6       9.7      44.8
```

---

## 3. 核心方法详解

### 3.1 数据读取 `read_data()`

```python
def read_data(self, file_path: str = '') -> None
```

**流程：**

1. 检查 `_is_data_loaded`，避免重复加载
2. 检测文件格式（CSV/Excel/TXT）
3. CSV：尝试 `utf-8-sig → gbk → gb2312 → utf-8 → latin-1` 多编码回退
4. Excel：读取第一个 sheet
5. TXT：尝试制表符分隔读取，多编码回退
6. 执行 `columns_preprocess(to_uppercase=True, remove_all_spaces=True)`
7. 调用 `_validate_data()` 完整性验证
8. 更新 `_is_data_loaded = True`

> ⚠️ 与早期版本对比：**不再计算分辨率**，直接跳过该步骤。

### 3.2 高级列名清理 `columns_preprocess()`

```python
def columns_preprocess(
    dataframe=None,
    remove_all_spaces=False,
    to_lowercase=False,
    to_uppercase=False,
    inplace=True
) -> pd.DataFrame
```

支持 4 种清理操作（可组合）：

| 选项 | 效果 |
|------|------|
| `remove_all_spaces=True` | 删除所有空格（包括中间空格） |
| `to_lowercase=True` | 转小写 |
| `to_uppercase=True` | 转大写 |
| `inplace=False` | 返回副本而非原地修改 |

**岩心数据实际调用：**
```python
self.columns_preprocess(self._data, to_uppercase=True, remove_all_spaces=True)
# 效果: 'DEPTH   ' → 'DEPTH'（矿物中文名称不受影响）
```

**算法（逐列处理）：**

1. 去除两端空格
2. 去除开头的下划线（`lstrip('_')`）
3. 根据 `remove_all_spaces` 删除/合并空格
4. 根据大小写选项转换
5. **检查列名唯一性**（重复则抛异常）

### 3.3 数据验证 `_validate_data()`

| 检查项 | 通过条件 | 失败时 |
|--------|----------|--------|
| 空数据表 | `not df.empty` | 抛出 `CoreException` |
| 空值统计 | 存在空值 | 警告日志（不阻断） |
| 深度单调性 | `np.diff(depths) > 0` 全为 True | 抛出 `CoreException` |
| 深度列名 | 第一列列名含 `DEPTH` | 警告日志（不阻断） |

### 3.4 数据获取 `get_data()`

```python
def get_data(
    curve_names: List[str] = None,
    depth_range: List[float] = None
) -> pd.DataFrame
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `curve_names` | `List[str]` | `None` | 指定列名，`None`→所有列 |
| `depth_range` | `List[float]` | `None` | 深度范围 `[min, max]`，`None`→不限制 |

**返回：** 筛选后的 `DataFrame`

### 3.5 数据摘要 `get_summary()`

```python
def get_summary() -> Dict[str, any]
```

**返回字段：**

```python
{
    'well_name': '姬119H2',
    'file_path': '...core.csv',
    'is_loaded': True,
    'data_shape': (34, 6),
    'curve_count': 6,
    'columns': ['DEPTH', '石英', '钾长石', ...],
    'depth_min': 2726.53,
    'depth_max': 2760.76,
    'depth_range': (2726.53, 2760.76),
    'sample_count': 34
}
```

> ⚠️ 与早期版本对比：**`resolution` 字段已移除**。

### 3.6 曲线名称获取 `get_curve_names()`

```python
def get_curve_names() -> List[str]
```

惰性加载 + 返回列名列表副本。

---

## 4. 日志系统

每个关键步骤均有日志输出：

| 日志级别 | 触发场景 |
|----------|----------|
| INFO | 文件格式检测、读取成功、列名修改数量、深度筛选结果 |
| WARNING | 文件不存在、发现空值列、深度列名不含 DEPTH |
| ERROR | 无法解码文件、列名重复、深度非单调 |

---

## 5. 与 `DataTable` 的对比

| 对比维度 | DataTable | DataCore |
|----------|-----------|----------|
| 数据来源 | 岩性类型表（解释结果） | 岩心实验（全岩分析） |
| 数值类型 | 整数类别标签 | 浮点数百分比 |
| 深度列 | 严格单调递增 | 严格单调递增 |
| 格式互转 | 2列↔3列 | 无（稀疏点数据） |
| 类型替换 | 有（标签映射） | 无（数值数据） |
| 分辨率 | 有（DataTable 保留） | 无（DataCore 已移除） |

---

## 6. 设计亮点

| 特性 | 实现方式 |
|------|----------|
| 多编码兼容 | CSV/TXT 读取尝试 5 种编码 |
| 惰性加载 | 所有数据访问方法在未加载时自动触发读取 |
| 深度验证 | 严格单调递增检查，保证数据质量 |
| 日志完善 | 每个关键步骤都有 INFO/WARNING/ERROR 分级日志 |
| 简洁设计 | 不存储分辨率，专注数据读取和筛选 |

---

## 7. 版本变更历史

### v2.0（2026-07-30）

- **移除** `_resolution` 属性
- **移除** `_calculate_resolution()` 方法
- **移除** `read_data()` 中的分辨率计算和日志
- **移除** `get_summary()` 返回字典中的 `resolution` 字段
- **原因**：稀疏岩心数据的分辨率概念不适用，数据读取聚焦于数据获取本身

---

*文档版本：2.0 | 对应源码版本：2026-07-30*
