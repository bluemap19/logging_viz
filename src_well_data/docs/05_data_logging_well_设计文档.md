# `data_logging_well.py` 设计文档

## 文件概述

**文件路径：** `src_well_data/data_logging_well.py`  
**核心类：** `DATA_WELL`  
**功能定位：** 井数据统一管理器，作为 Facade（门面）模式整合 DataLogging、DataFMI、DataNMR、DataTable、DataCore 五个子模块，提供统一的井数据访问接口。  
**代码规模：** ~650 行（含测试代码）  
**最后更新：** 2026-07-31

---

## 1. 设计模式：Facade（门面模式）

`DATA_WELL` 不直接持有数据，而是作为统一入口，按需创建和调用各子模块，对外屏蔽底层复杂度。

```
用户代码
    │
    ▼
DATA_WELL (Facade)
    │
    ├─► DataLogging   — 常规测井曲线
    ├─► DataFMI       — 电成像数据
    ├─► DataNMR       — 核磁共振数据
    ├─► DataTable     — 岩性类型表
    └─► DataCore      — 岩心实验数据
```

---

## 2. 类架构

### 2.1 数据容器

```python
self.logging_dict: Dict[str, DataLogging] = {}  # path → DataLogging实例
self.table_dict: Dict[str, DataTable] = {}      # path → DataTable实例
self.FMI_dict: Dict[str, DataFMI] = {}          # path → DataFMI实例
self.NMR_dict: Dict[str, Any] = {}              # path → DataNMR实例
self.core_dict: Dict[str, DataCore] = {}        # path → DataCore实例
```

### 2.2 路径容器

```python
self.path_list_logging: List[str] = []  # 所有测井文件路径
self.path_list_table: List[str] = []    # 所有表格文件路径
self.path_list_fmi: List[str] = []      # 所有 FMI 文件路径
self.path_list_nmr: List[str] = []      # 所有 NMR 文件路径
self.path_list_core: List[str] = []     # 所有岩心数据文件路径
```

### 2.3 文件识别关键字

| 容器 | 关键字 | 文件类型 | 扩展名 |
|------|--------|----------|--------|
| `LOGGING_KW` | `['logging']` | 常规测井 | `.xlsx`, `.csv` |
| `TABLE_KW` | `['table', 'LITHO_TYPE']` | 岩性表 | `.xlsx`, `.csv` |
| `FMI_KW` | `['DYNA', 'STAT']` | 电成像 | `.txt` |
| `NMR_KW` | `['NMR']` | 核磁 | `.csv`, `.txt` |
| `CORE_KW` | `['core']` | 岩心实验 | `.csv`, `.xlsx`, `.txt` |

---

## 3. 核心方法

### 3.1 初始化 `__init__()`

```python
def __init__(self, path_folder: str = '', WELL_NAME: str = ''):
    self.well_path = path_folder
    self.WELL_NAME = WELL_NAME or os.path.basename(path_folder)
    self.scan_files()  # 自动扫描识别所有类型的文件
```

### 3.2 文件扫描 `scan_files()`

调用外部模块：`src_file_op.dir_operation.search_files_by_criteria`

对 5 类文件分别扫描，结果填充到对应 `path_list_*` 容器。

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
def init_*(self, path: str = ''):
    if not path: path = self.path_list_*[0]  # 默认取第一个
    if path not in self.*_dict:
        self.*_dict[path] = Data*(path=path, well_name=self.WELL_NAME)
```

**模式特点：**

- **惰性初始化**：文件路径不在字典中才创建实例
- **单例**：同一路径不会重复创建实例
- **默认行为**：不指定路径时取第一个扫描到的文件

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

#### `get_core()`

```python
def get_core(
    key: str = '',
    curve_names: Optional[List[str]] = None,
    depth_range: Optional[List[float]] = None
) -> pd.DataFrame
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `key` | `str` | `''` | 文件路径或关键字，'' → 取第一个扫描到的文件 |
| `curve_names` | `List[str]` | `None` | 指定列名，None → 获取所有列 |
| `depth_range` | `List[float]` | `None` | 深度范围 `[min, max]`，None → 不限制 |

**返回：** 包含指定列和深度范围的岩心实验数据 DataFrame

**典型列名：** `['DEPTH', '石英', '钾长石', '斜长石', '黄铁矿', '黏土矿物']`

**数据特点：** 稀疏采样（间隔约 1m），远大于测井分辨率。

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
    tolerance=0,
    depth_limit: Optional[List[float]] = None,
) -> pd.DataFrame
```

**流程：**

1. 获取测井曲线数据 `get_logging()`
2. 获取岩性表数据 `get_table_2_replaced()`
3. 按深度列排序
4. 调用 `src_logging.logging_combine.combine_logging_table()` 执行合并
5. 删除 NaN，类型列转为 int
6. 可选重命名类型列

#### `combine_logging_core()` ⭐ 新增

**功能：** 将测井曲线数据与岩心实验数据合并。

```python
def combine_logging_core(
    logging_key: str = '',
    curve_names_logging: Optional[List[str]] = None,
    norm: bool = False,
    tolerance: float = 0,
    core_key: str = '',
    curve_names_core: Optional[List[str]] = None,
    depth_limit: Optional[List[float]] = None,
) -> pd.DataFrame
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `logging_key` | `str` | `''` | 测井文件路径或关键字 |
| `curve_names_logging` | `List[str]` | `None` | 要保留的 logging 曲线列名，None→全部 |
| `norm` | `bool` | `False` | 是否对测井曲线归一化 |
| `tolerance` | `float` | `0` | 深度合并容差（米） |
| `core_key` | `str` | `''` | 岩心文件路径或关键字，''→跳过 |
| `curve_names_core` | `List[str]` | `None` | 要保留的 core 列名，None→全部 |
| `depth_limit` | `List[float]` | `None` | 全局深度限制 [min, max] |

**流程：**

1. 获取 logging 数据作为主基准
2. 获取 core 数据（稀疏采样）
3. 调用 `combine_logging_table()` 执行深度最近邻匹配
4. 无匹配处保留 NaN（不删除行）

**返回：** `depth + logging_curves + core_curves` 格式 DataFrame

#### `combine_logging_table_core()` ⭐ 新增

**功能：** 将 logging、table、core 三类数据按深度最近邻合并。

```python
def combine_logging_table_core(
    logging_key: str = '',
    curve_names_logging: Optional[List[str]] = None,
    table_key: str = '',
    replace_dict: Optional[Dict] = None,
    new_col: str = 'Type',
    norm: bool = False,
    tolerance: float = 0,
    core_key: str = '',
    curve_names_core: Optional[List[str]] = None,
    depth_limit: Optional[List[float]] = None,
) -> pd.DataFrame
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `logging_key` | `str` | `''` | 测井文件路径或关键字 |
| `curve_names_logging` | `List[str]` | `None` | 要保留的 logging 曲线列名 |
| `table_key` | `str` | `''` | 类型表路径或关键字，''→跳过 |
| `replace_dict` | `Dict` | `None` | 类型替换字典 |
| `new_col` | `str` | `'Type'` | 替换后的标签列名 |
| `norm` | `bool` | `False` | 是否归一化 |
| `tolerance` | `float` | `0` | 深度容差（米） |
| `core_key` | `str` | `''` | 岩心路径或关键字，''→跳过 |
| `curve_names_core` | `List[str]` | `None` | 要保留的 core 列名 |
| `depth_limit` | `List[float]` | `None` | 全局深度限制 |

**合并策略：**

1. 获取 logging 数据（主基准）
2. 若 `table_key` 非空：调用 `combine_logging_table()`
3. 若 `core_key` 非空：调用 `combine_logging_core()`
4. 若两者均存在：`pd.merge(on=key_cols, how='outer')`
5. 若两者均不存在：退化为普通 logging 获取

**返回：** `depth + logging_curves + table_label + core_curves` 格式 DataFrame

---

## 4. 路径获取接口

| 方法 | 说明 |
|------|------|
| `get_path_list_logging()` | 获取测井文件路径列表 |
| `get_path_list_fmi()` | 获取 FMI 文件路径列表 |
| `get_path_list_nmr()` | 获取 NMR 文件路径列表 |
| `get_path_list_table()` | 获取表格文件路径列表 |
| `get_path_list_core()` | 获取岩心数据文件路径列表 |
| `search_file_path_list(name_keywords, file_extensions)` | 按关键字精确搜索（AND 匹配） |

**注意：** 原 `search_*_path_list()` 系列方法已被统一的 `search_file_path_list()` 替代。

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
        "paths_core": self.path_list_core,
        "logging_files_num": len(self.path_list_logging),
        "fmi_files_num": len(self.path_list_fmi),
        "table_files_num": len(self.path_list_table),
        "nmr_files_num": len(self.path_list_nmr),
        "core_files_num": len(self.path_list_core),
    }
```

---

## 6. 已知问题状态

### 6.1 `init_NMR` Bug（✅ 已修复）

```python
# 修复前（Bug）
def init_NMR(self, path: str = ''):
    if not path:
        if not self.path_list_fmi:          # ← 错误：使用了 path_list_fmi
            return
        path = self.path_list_fmi[0]        # ← 错误：使用了 path_list_fmi

# 修复后
def init_NMR(self, path: str = ''):
    if not path:
        if not self.path_list_nmr:          # ← 修复：改为 path_list_nmr
            return
        path = self.path_list_nmr[0]        # ← 修复：改为 path_list_nmr
```

### 6.2 其他说明

| 项目 | 说明 |
|------|------|
| `_get_default_obj` | 依赖 Python 3.7+ 有序字典保证 |
| 文件扫描 | 依赖关键字匹配，文件名需符合命名规则 |
| 合并容差 | 默认 tolerance=0，需根据数据分辨率调整 |

---

## 7. 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-07-30 | 初始版本（不含 DataCore） |
| v2.0 | 2026-07-30 | 新增 DataCore 集成，修复 init_NMR Bug |
| v2.1 | 2026-07-31 | 新增 `combine_logging_core()`、`combine_logging_table_core()` 方法；简化 `search_*_path_list` 为统一接口 |

---

*文档版本：2.1 | 对应源码版本：2026-07-31*
