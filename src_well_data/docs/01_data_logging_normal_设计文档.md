# `data_logging_normal.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_normal.py`  
**核心类：** `DataLogging`  
**功能定位：** 常规测井曲线数据管理，支持多格式读取、曲线名称映射、数据归一化和分辨率计算。  
**代码规模：** ~380 行（不含测试代码）

---

## 1. 类架构

### 1.1 异常类

```
DataLoggingException(Exception)
```

用于测井数据读取、映射、归一化等环节的异常捕获。

### 1.2 文件格式枚举

```python
class FileFormat(Enum):
    CSV = '.csv'
    EXCEL = '.xlsx'
    TEXT = '.txt'
    UNKNOWN = 'unknown'
```

### 1.3 核心类：`DataLogging`

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_data` | `pd.DataFrame` | 原始测井数据 |
| `_data_normed` | `pd.DataFrame` | 归一化后的测井数据 |
| `_data_with_type` | `pd.DataFrame` | 测井数据与岩性类型合并结果 |
| `_data_normed_with_type` | `pd.DataFrame` | 归一化+岩性合并结果 |
| `_curve_names` | `List[str]` | 曲线名称列表 |
| `_file_path` | `str` | 数据文件路径 |
| `_logging_name` | `str` | 井名标识 |
| `_resolution` | `float` | 深度分辨率（米/点） |
| `mapping_dict` | `Dict[str, List[str]]` | 曲线名称别名映射字典 |
| `_is_data_loaded` | `bool` | 数据加载状态 |

#### 类常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `DEFAULT_RESOLUTION` | `-1.0` | 未计算状态标记 |
| `CONFIG_FILE_NAME` | `"COLS_MAPPING.xml"` | 映射配置文件名 |
| `DEFAULT_CONFIG_PATH` | `r"D:\GitHub\zfhlog\src_well_data"` | 配置文件默认路径 |

---

## 2. 核心方法

### 2.1 数据读取 `read_data()`

**流程：**

1. 检测数据是否已加载（避免重复）
2. 检测文件格式（CSV/Excel/TXT）
3. CSV：尝试 `utf-8-sig → gbk → utf-8 → latin-1` 多编码回退
4. Excel：读取第一个 sheet
5. TXT：跳过前 8 行（LAS 头），按制表符分隔
6. 执行 `columns_preprocess()` — 去除空格、大写化
7. 计算分辨率 `_calculate_resolution()`
8. 更新 `_is_data_loaded = True`

**异常处理：** 任何环节失败均抛出 `DataLoggingException`。

### 2.2 高级列名清理 `columns_preprocess()`

支持 4 种清理操作（可组合）：

```python
def columns_preprocess(
    dataframe=None,
    remove_all_spaces=False,   # 删除所有空格（包括中间）
    to_lowercase=False,        # 转小写
    to_uppercase=False,        # 转大写
    inplace=True
)
```

**算法：**

1. 去除两端空格
2. **去除开头连续下划线**（`lstrip('_')`）— 这是关键细节
3. 处理空格（删除全部 / 合并连续空格）
4. 大小写转换
5. 检查列名唯一性（重复则抛异常）
6. 如果处理的是 `self._data`，同步更新 `_curve_names`

### 2.3 曲线名称映射 `input_cols_mapping()`

**功能：** 将别名映射为标准曲线名。

**示例：**
```
输入: ['GRC', 'CN', 'DT24']
目标: ['Depth', 'GR', 'CNL', 'AC']
映射: GRC→GR, CN→CNL, DT24→AC
输出: ['Depth', 'GR', 'CNL', 'AC']
```

**算法（关键步骤）：**

1. 复制输入列表
2. 找出不在目标列表中的曲线（需要映射的）
3. **从后向前** 遍历需要映射的曲线（避免索引变化）
4. 在 `mapping_dict` 中查找别名对应标准名
5. 用集合交集确认标准名在目标列表中存在
6. 替换别名为标准名称
7. 剩余无法映射的曲线 → 抛异常

**为什么要从后向前？** 因为正向遍历时 `list.index()` 返回第一个匹配位置，一旦替换某元素，后续索引会错位。从后向前则不受影响。

### 2.4 分辨率计算 `_calculate_resolution()`

**算法：**

1. 提取第一列（深度列）
2. 计算相邻深度差 `np.diff(depth_array)`
3. 使用众数 `find_mode()` 作为分辨率

### 2.5 数据获取 `get_data()`

**流程：**

1. 惰性加载（数据未加载则调用 `read_data()`）
2. 确定曲线列表（None → 所有曲线）
3. 确保深度列在结果中
4. 曲线名称映射
5. 可选：按 `depth_delete` 列表删除指定深度点

### 2.6 归一化数据获取 `get_data_normed()`

**惰性策略：**

- `_data_normed` 为空 → 重新归一化
- 指定了曲线但这些曲线不在 `_data_normed` 中 → 重新归一化
- 否则直接返回已缓存的归一化数据

**调用外部模块：** `src_logging.curve_preprocess.data_Normalized`

---

## 3. 配置文件格式 `COLS_MAPPING.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<LogAliasMapping>
    <LogType name="depth">
        <Alias>#Depth</Alias>
        <Alias>#DEPTH</Alias>
        ...
    </LogType>
    <LogType name="GR">
        <Alias>GRC</Alias>
        ...
    </LogType>
    ...
</LogAliasMapping>
```

---

## 4. 设计亮点

| 特性 | 实现方式 |
|------|----------|
| 多编码兼容 | CSV 读取尝试 4 种编码 |
| 惰性加载 | 所有数据访问方法在数据未加载时自动触发读取 |
| 配置外部化 | 曲线映射通过 XML 文件配置，支持扩展 |
| 索引安全 | 曲线映射从后向前遍历避免索引偏移 |
| 列名标准化 | 去除空格、大小写统一、去开头下划线 |

---

## 5. 潜在问题

1. **`TXT 文件处理过于简化：** 直接 `np.loadtxt(file_path, skiprows=8)`，未考虑实际 LAS 格式的多样性
2. **`_data_with_type` 和 `_data_normed_with_type` 属性已定义但未被任何方法使用**
3. **`get_summary()` 返回的字段在数据未加载时可能不完整**

---

*文档版本：1.0 | 对应源码版本：2026-07-28*
