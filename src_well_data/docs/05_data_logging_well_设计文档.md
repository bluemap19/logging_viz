# `data_logging_well.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_well.py`  
**核心类：** `DATA_WELL`  
**功能定位：** 井数据统一管理器，作为 Facade（门面）模式整合 DataLogging、DataFMI、DataNMR、DataTable 四个子模块，提供统一的井数据访问接口。  
**代码规模：** ~450 行（不含测试代码）

---

## 1. 设计模式：Facade（门面模式）

`DATA_WELL` 不直接持有数据，而是作为统一入口，按需创建和调用各子模块（DataLogging、DataFMI、DataNMR、DataTable），对外屏蔽底层复杂度。

```
用户代码
    │
    ▼
DATA_WELL (Facade)
    │
    ├─► DataLogging   — 常规测井曲线
    ├─► DataFMI       — 电成像数据
    ├─► DataNMR       — 核磁共振数据
    └─► DataTable     — 岩性类型表
```

---

## 2. 类架构

### 2.1 数据容器

```python
self.logging_dict: Dict[str, DataLogging] = {}  # path → DataLogging实例
self.table_dict: Dict[str, DataTable] = {}      # path → DataTable实例
self.FMI_dict: Dict[str, DataFMI] = {}          # path → DataFMI实例
self.NMR_dict: Dict[str, DataNMR] = {}          # path → DataNMR实例
```

### 2.2 路径容器

```python
self.path_list_logging: List[str] = []  # 所有测井文件路径
self.path_list_table: List[str] = []    # 所有表格文件路径
self.path_list_fmi: List[str] = []      # 所有 FMI 文件路径
self.path_list_nmr: List[str] = []      # 所有 NMR 文件路径
```

### 2.3 文件识别关键字

| 容器 | 关键字 | 文件类型 | 扩展名 |
|------|--------|----------|--------|
| `LOGGING_KW` | `['logging']` | 常规测井 | `.xlsx`, `.csv` |
| `TABLE_KW` | `['table', 'LITHO_TYPE']` | 岩性表 | `.xlsx`, `.csv` |
| `FMI_KW` | `['DYNA', 'STAT']` | 电成像 | `.txt` |
| `NMR_KW` | `['NMR']` | 核磁 | `.csv`, `.txt` |

---

## 3. 核心方法

### 3.1 初始化 `__init__()`

```python
def __init__(self, path_folder: str = '', WELL_NAME: str = ''):
    self.well_path = path_folder
    self.WELL_NAME = WELL_NAME or os.path.basename(path_folder)
    self.scan_files()  # 自动扫描识别文件
```

### 3.2 文件扫描 `scan_files()`

调用外部模块：`src_file_op.dir_operation.search_files_by_criteria`

对 4 类文件分别扫描，结果填充到对应 `path_list_*` 容器。

**关键参数：**

```python
search_files_by_criteria(
    path,                    # 搜索目录
    name_keywords=[...],     # 文件名需包含的关键字
    file_extensions=[...],   # 扩展名过滤
    all_keywords=False       # 关键字匹配逻辑：任一包含(OR)
)
```

### 3.3 子模块初始化（惰性单例模式）

```python
def init_logging(self, path: str = ''):
    if not path: path = self.path_list_logging[0]  # 默认取第一个
    if path not in self.logging_dict:
        self.logging_dict[path] = DataLogging(path=path, well_name=self.WELL_NAME)
```

**模式特点：**

- 惰性初始化：文件路径不在字典中才创建实例
- 单例：同一路径不会重复创建实例
- 默认行为：不指定路径时取第一个扫描到的文件

### 3.4 统一访问接口

#### `get_logging()`

```python
def get_logging(
    key='',
    curve_names: List[str] = None,
    norm: bool = False,
    depth_limit: List[float] = []
) -> pd.DataFrame
```

| 参数 | 说明 |
|------|------|
| `key` | 文件路径或关键字（模糊匹配） |
| `curve_names` | 指定曲线列表，None→所有曲线 |
| `norm` | True→返回归一化数据 |
| `depth_limit` | 深度范围 [min, max] |

**流程：**

1. `init_logging(key)` — 惰性初始化
2. `obj.get_data_normed()` 或 `obj.get_data()` — 获取数据
3. `process_depth_segment()` — 深度筛选（调用外部模块）

#### `get_table()`

```python
def get_table(
    key='',
    mode='3',          # '3' → 3列格式, '2' → 2列格式
    replaced=False,
    replace_dict=None,
    new_col='Type_Replaced'
) -> pd.DataFrame
```

#### `get_FMI()` / `get_NMR()`

```python
def get_FMI(key='', depth=None) -> Tuple[np.ndarray, np.ndarray]
def get_NMR(key='', depth=None) -> Tuple[np.ndarray, np.ndarray]
```

返回 `(data, depth)` 元组。

#### `get_FMI_texture()`

```python
def get_FMI_texture(key='', texture_config=None) -> pd.DataFrame
```

调用 `DataFMI.get_texture()`，返回纹理特征 DataFrame。

#### `get_FMI_textures()`

```python
def get_FMI_textures(texture_config, path_config={}) -> pd.DataFrame
```

**功能：** 合并动态成像（DYNA）和静态成像（STAT）的纹理特征。

**流程：**

1. 检查缓存文件 `{well_name}_texture_logging_{windows_length}.csv`
2. 存在则直接读取
3. 不存在则分别计算 DYNA 和 STAT 纹理
4. 横向拼接 `pd.concat([texture_stat, texture_dyna.iloc[:, 1:]], axis=1)`
5. 保存到缓存文件

#### `get_FMI_fde()` / `get_FMI_fdes()`

- `get_FMI_fde()`：获取指定路径的 FDE 数据
- `get_FMI_fdes()`：获取 DYNA 和 STAT 两个 FDE 数据

#### `combine_logging_table()`

**核心方法：** 将测井曲线数据与岩性类型表合并。

```python
def combine_logging_table(
    logging_key='',
    curve_names_logging=None,
    table_key='',
    replace_dict=None,
    new_col='Type',
    norm=False,
    tolerance=0.5,  # 深度容差（米）
) -> pd.DataFrame
```

**流程：**

1. 获取测井曲线数据 `get_logging()`
2. 获取岩性表数据 `get_table_2_replaced()`
3. 按深度列排序
4. 调用 `src_logging.logging_combine.combine_logging_table()` 执行合并
5. 删除 NaN，类型列转为 int
6. 可选重命名类型列

---

## 4. 搜索辅助方法

| 方法 | 说明 |
|------|------|
| `search_logging_path_list(new_kw=[])` | 用新关键字搜索测井文件（all_keywords=True） |
| `search_table_path_list(new_kw=[])` | 用新关键字搜索表格文件 |
| `search_fmi_path_list(new_kw=[])` | 用新关键字搜索 FMI 文件 |
| `search_nmr_path_list(new_kw=[])` | 用新关键字搜索 NMR 文件 |
| `search_data_path(keywords, path_list)` | 在指定路径列表中搜索 |

---

## 5. 数据概览 `well_summary()`

```python
def well_summary(self) -> Dict[str, Any]:
    return {
        "well": self.WELL_NAME,
        "path": self.well_path,
        "paths_logging": self.path_list_logging,
        "paths_fmi": self.path_list_fmi,
        "paths_table": self.path_list_table,
        "paths_nmr": self.path_list_nmr,
        "logging_files_num": len(self.path_list_logging),
        "fmi_files_num": len(self.path_list_fmi),
        "table_files_num": len(self.path_list_table),
        "nmr_files_num": len(self.path_list_nmr),
    }
```

---

## 6. ⚠️ 严重 Bug：`init_NMR` 方法

```python
def init_NMR(self, path: str = ''):
    if not path:
        if not self.path_list_fmi:  # ← 错误！应该是 path_list_nmr
            return
        path = self.path_list_fmi[0]  # ← 错误！应该是 path_list_nmr
```

**问题：** `init_NMR` 方法错误地使用了 `path_list_fmi` 而非 `path_list_nmr`。

**后果：** 如果 NMR 文件路径列表不为空但 FMI 文件列表为空，NMR 初始化会返回空列表；即使返回了非空值，也取的是 FMI 文件路径而非 NMR 文件路径。

---

## 7. 潜在问题

1. **⚠️ `init_NMR` Bug**（如上所述）
2. **`_get_default_obj` 使用 `next(iter(dict.values()))`**：字典遍历顺序在 Python 3.7+ 虽然有序，但逻辑上不够明确
3. **文件扫描依赖关键字匹配**：如果文件名不符合关键字规则，会被遗漏
4. **`combine_logging_table` 中 `tolerance` 参数未暴露其含义**：0.5 米容差可能不适合高精度场景

---

*文档版本：1.0 | 对应源码版本：2026-07-30*
