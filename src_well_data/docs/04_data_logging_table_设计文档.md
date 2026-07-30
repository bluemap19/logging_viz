# `data_logging_table.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_table.py`  
**核心类：** `DataTable`  
**功能定位：** 测井岩性类型表管理，支持 2 列（深度-类型）和 3 列（开始-结束-类型）格式的读取、互转和类型替换。  
**代码规模：** ~450 行（不含测试代码）

---

## 1. 类架构

### 1.1 异常类与枚举

```python
class DataTableException(Exception)

class TableFormat(Enum):
    UNKNOWN = 0          # 未知
    DEPTH_TYPE = 1       # 深度-类型 (n×2)
    START_END_TYPE = 2   # 开始-结束-类型 (n×3)
```

### 1.2 核心类：`DataTable`

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_table_2` | `pd.DataFrame` | 深度-类型格式原始数据 |
| `_table_2_replaced` | `pd.DataFrame` | 类型替换后的 2 列数据 |
| `_table_3` | `pd.DataFrame` | 开始-结束-类型格式原始数据 |
| `_table_3_replaced` | `pd.DataFrame` | 类型替换后的 3 列数据 |
| `_table_resolution` | `float` | 深度采样分辨率（米） |
| `_file_path` | `str` | 数据文件路径 |
| `_well_name` | `str` | 井名标识 |
| `_raw_data` | `pd.DataFrame` | 从文件直接读取的原始数据 |
| `_replace_dict` | `Dict` | 类型替换映射字典 |
| `_is_data_loaded` | `bool` | 数据加载状态 |

#### 列名常量

```python
COLUMN_NAMES_2 = ['Depth', 'Type']           # 2列表格列名
COLUMN_NAMES_3 = ['Depth_Start', 'Depth_End', 'Type']  # 3列表格列名
```

---

## 2. 两种数据格式

### 2.1 深度-类型格式 (2列)

| Depth | Type |
|-------|------|
| 1000.0 | 1 |
| 1000.5 | 1 |
| 1001.0 | 2 |
| 1001.5 | 2 |
| 1002.0 | 3 |
| ... | ... |

**特点：** 每个深度点一行，深度严格单调递增，适合与测井曲线直接对齐。

### 2.2 开始-结束-类型格式 (3列)

| Depth_Start | Depth_End | Type |
|-------------|-----------|------|
| 1000.0 | 1001.0 | 1 |
| 1001.0 | 1002.5 | 2 |
| 1002.5 | 1005.0 | 3 |
| ... | ... | ... |

**特点：** 每个连续岩性段一行，存储效率高，适合岩性解释结果输出。

---

## 3. 核心方法

### 3.1 格式自动检测 `_detect_table_format()`

```python
def _detect_table_format(data: pd.DataFrame) -> TableFormat:
    if data.shape[1] == 2: return DEPTH_TYPE
    elif data.shape[1] == 3: return START_END_TYPE
    elif data.shape[1] >= 4: return START_END_TYPE  # 取前3列
    else: raise DataTableException(...)
```

### 3.2 数据读取 `read_data()`

**主流程：**

1. 检查 `_is_data_loaded`，避免重复加载
2. 调用 `_read_file()` 读取原始数据
3. 调用 `_process_data_format()` 处理格式
4. 调用 `_extract_replace_dict()` 提取替换字典
5. 更新 `_is_data_loaded = True`

**CSV 多编码回退：** `utf-8-sig → gbk → utf-8 → latin-1`

### 3.3 格式处理 `_process_data_format()`

根据检测到的格式执行不同逻辑：

**DEPTH_TYPE (2列)：**

```python
self._table_2 = raw_data.iloc[:, :2]  # 取前2列
self._table_2.columns = COLUMN_NAMES_2
self._check_table_2()                  # 完整性检查
self._table_resolution = 自动计算    # 调用 get_resolution_by_depth
self._convert_2_to_3()                 # 惰性转换为3列
```

**START_END_TYPE (3列)：**

```python
self._table_3 = raw_data.iloc[:, :3]  # 取前3列
self._table_3.columns = COLUMN_NAMES_3
self._check_table_3()                  # 完整性检查
if self._table_resolution <= 0:
    self._table_resolution = 0.1      # 使用默认值
self._convert_3_to_2()                 # 惰性转换为2列
```

### 3.4 数据完整性检查

#### `_check_table_2()`

1. 数据表是否为空
2. 列数是否等于 2
3. 是否存在空值（自动丢弃并警告）
4. **深度值是否严格单调递增**（非递增则抛异常）

#### `_check_table_3()`

1. 数据表是否为空
2. 列数是否等于 3
3. 是否存在空值（自动丢弃并警告）
4. **开始深度 < 结束深度**（非法则抛异常）
5. 相邻区间是否连续（不连续则警告，不阻断）

### 3.5 格式互转

#### `_convert_2_to_3()` — 2列→3列

调用外部模块：`src_table.table_process.table_2_to_3`

**算法：** 相邻同类型点合并为区间段，边界取中点。

#### `_convert_3_to_2()` — 3列→2列

调用外部模块：`src_table.table_process.table_3_to_2`

**参数：**

```python
def _convert_3_to_2(self, resolution: Optional[float] = None):
    resolution = resolution or self._table_resolution
    table_2_array = table_3_to_2(self._table_3.values, step=resolution)
```

**算法：** 按分辨率对每个区间进行采样，生成深度点序列。

### 3.6 类型替换 `_apply_type_replacement()`

```python
def _apply_type_replacement(
    replace_dict: Optional[Dict] = None,
    new_col: str = 'Type_Replaced'
):
    replace_dict = replace_dict or self._replace_dict

    # 对 _table_2: 复制 + 添加新列
    self._table_2_replaced = self._table_2.copy()
    self._table_2_replaced[new_col] = self._table_2.iloc[:, -1].map(
        lambda x: replace_dict.get(x, x)  # 找不到则保留原值
    )

    # 对 _table_3: 同理
    self._table_3_replaced = self._table_3.copy()
    self._table_3_replaced[new_col] = self._table_3.iloc[:, -1].map(...)
```

### 3.7 替换字典提取 `_extract_replace_dict()`

调用外部模块：`src_table.table_process.get_replace_dict`

从原始数据的最后一列（类型列）提取唯一值生成映射。

---

## 4. 数据获取接口

| 方法 | 返回格式 | 说明 |
|------|----------|------|
| `get_table_2()` | DataFrame (2列) | 原始 2 列数据 |
| `get_table_2_replaced()` | DataFrame (2列) | 类型替换后的 2 列 |
| `get_table_3()` | DataFrame (3列) | 原始 3 列数据 |
| `get_table_3_replaced()` | DataFrame (3列) | 类型替换后的 3 列 |

所有方法均遵循惰性策略：未加载则读取，未转换则转换。

---

## 5. 分辨率管理

```python
def set_resolution(self, resolution: float) -> None:
    if resolution <= 0: raise DataTableException(...)
    self._table_resolution = resolution
    if not self._table_3.empty:
        self._convert_3_to_2(resolution)  # 分辨率改变后重新转换为2列
```

---

## 6. 潜在问题

1. **`_replace_dict` 仅从最后一列提取**，若原始数据有 4+ 列且类型不在最后一列，会遗漏
2. **`_table_resolution` 默认值 0.1 可能不适用于高分辨率数据**（测井数据通常 0.0025）
3. **2→3 转换后 3→2 转换循环依赖风险**：虽然有 `if self._table_2.empty` 保护，但若两次转换都执行且分辨率不同，可能导致数据不一致

---

*文档版本：1.0 | 对应源码版本：2026-07-28*
