# 🗂️ 井群 Project 数据管理模块

> Well Project Data Management Module

**版本：** v1.0  
**最后更新：** 2026-03-14  
**关联源文件：** `src_well_project/LOGGING_PROJECT.py`

---

## 一、模块概述

### 1.1 模块定位

井群 Project 数据管理模块是整个测井数据可视化系统的**最高管理层**，负责多井项目的统一管理和数据整合。它封装了多个单井数据对象 (DATA_WELL)，提供跨井数据查询、合并、对比和分析的高级接口，是进行多井对比、区域地质分析和批量数据处理的核心模块。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| **项目管理** | 管理多井数据，维护井名 - 路径映射 |
| **跨井查询** | 按井名或文件特征获取数据 |
| **数据合并** | 合并多井测井数据，支持垂直链接 |
| **批量处理** | 批量计算 FMI 纹理等处理任务 |
| **文件搜索** | 跨井搜索特定类型的文件 |

### 1.3 技术特点

- 统一的单井对象管理
- 灵活的数据获取方式
- 支持多井数据垂直合并
- 批量处理优化
- 懒加载策略继承

### 1.4 系统架构位置

```
┌─────────────────────────────────────────┐
│      应用层 (可视化/分析/导出)           │
└─────────────────┬───────────────────────┘
                  │ 使用
                  ↓
┌─────────────────────────────────────────┐
│   项目层 (LOGGING_PROJECT) ← 本模块     │
│  - 管理多井数据                          │
│  - 井名、路径、数据字典                  │
│  - 数据合并、垂直链接                    │
└─────────────────┬───────────────────────┘
                  │ 封装
                  ↓
┌─────────────────────────────────────────┐
│      单井层 (DATA_WELL × N)              │
│  - 单井数据统一管理                      │
└─────────────────────────────────────────┘
```

---

## 二、类结构设计

### 2.1 LOGGING_PROJECT 类结构

```python
class LOGGING_PROJECT:
    """多井项目管理类"""
    
    def __init__(self, project_path: str):
        """
        初始化项目对象
        
        参数:
            project_path: 项目根路径
        """
        self.PROJECT_PATH = project_path
        self.WELL_NAMES: List[str] = []
        self.WELL_PATH: Dict[str, str] = {}
        self.WELL_DATA: Dict[str, DATA_WELL] = {}
        self.data_vc: pd.DataFrame = None  # 垂直合并后的总数据
        
        # 初始化井路径
        self.init_well_path()
    
    def init_well_path(self) -> None:
        """初始化井路径映射"""
        pass
    
    def get_well_data(self, well_name: str, file_path: str = '',
                      curve_names: List[str] = None, Norm: bool = False) -> pd.DataFrame:
        """
        获取单井测井数据
        
        参数:
            well_name: 井名
            file_path: 文件路径关键字
            curve_names: 曲线名称列表
            Norm: 是否归一化
        
        返回:
            测井数据 DataFrame
        """
        pass
    
    def get_well_data_by_charters(self, well_name: str,
                                   target_path_feature: List[str],
                                   target_file_type: str,
                                   curve_names: List[str] = None,
                                   Norm: bool = False) -> pd.DataFrame:
        """
        按特征获取单井数据
        
        参数:
            well_name: 井名
            target_path_feature: 路径特征关键字列表
            target_file_type: 文件类型 ('logging', 'table', 'fmi', 'nmr')
            curve_names: 曲线名称列表
            Norm: 是否归一化
        
        返回:
            测井数据 DataFrame
        """
        pass
    
    def search_target_file_path(self, well_name: str,
                                 target_path_feature: List[str],
                                 target_file_type: str) -> List[str]:
        """
        搜索目标文件路径
        
        参数:
            well_name: 井名
            target_path_feature: 路径特征关键字列表
            target_file_type: 文件类型
        
        返回:
            匹配的文件路径列表
        """
        pass
    
    def combined_all_logging_with_type(self,
                                        well_names: List[str],
                                        file_path_logging: str,
                                        file_path_table: str,
                                        curve_names_logging: List[str],
                                        curve_names_type: List[str],
                                        replace_dict: Dict,
                                        type_new_col: str,
                                        Norm: bool = False) -> pd.DataFrame:
        """
        合并测井数据与表格数据 (多井)
        
        参数:
            well_names: 井名列表
            file_path_logging: 测井文件路径关键字
            file_path_table: 表格文件路径关键字
            curve_names_logging: 测井曲线名称列表
            curve_names_type: 表格列名列表
            replace_dict: 类别替换字典
            type_new_col: 新列名
            Norm: 是否归一化
        
        返回:
            合并后的 DataFrame (包含 Well 列)
        """
        pass
    
    def get_fmi_texture(self, well_names: List[str],
                        file_path_logging: str,
                        Mode: str = 'MEAN',
                        texture_config: Dict = None) -> pd.DataFrame:
        """
        批量获取多井 FMI 纹理数据
        
        参数:
            well_names: 井名列表
            file_path_logging: 测井文件路径关键字
            Mode: 纹理模式 ('MEAN', 'SUB', 'X', 'Y')
            texture_config: 纹理计算配置
        
        返回:
            纹理数据 DataFrame (包含 Well 列)
        """
        pass
    
    def vertical_concat_wells(self, well_names: List[str], **kwargs) -> pd.DataFrame:
        """
        垂直链接多井数据
        
        参数:
            well_names: 井名列表
            **kwargs: 传递给 get_well_data 的参数
        
        返回:
            垂直合并后的 DataFrame
        """
        pass
```

### 2.2 核心属性详解

| 属性 | 类型 | 说明 |
|------|------|------|
| `PROJECT_PATH` | str | 项目根路径 |
| `WELL_NAMES` | List[str] | 井名列表 |
| `WELL_PATH` | Dict[str, str] | 井名→路径映射字典 |
| `WELL_DATA` | Dict[str, DATA_WELL] | 井名→DATA_WELL 对象映射 |
| `data_vc` | pd.DataFrame | 垂直合并后的总数据 (缓存) |

---

## 三、核心方法详解

### 3.1 初始化 (init_well_path)

**目的：** 扫描项目根路径，识别所有井文件夹并建立映射

**原理：**
1. 遍历项目根路径下的所有一级子文件夹
2. 每个子文件夹视为一口井
3. 子文件夹名作为井名，完整路径作为井路径
4. 为每口井创建 DATA_WELL 对象

**代码示例：**
```python
def init_well_path(self):
    """初始化井路径映射"""
    import os
    
    # 扫描项目根路径下的所有一级子文件夹
    for item in os.listdir(self.PROJECT_PATH):
        item_path = os.path.join(self.PROJECT_PATH, item)
        
        # 只处理文件夹
        if os.path.isdir(item_path):
            well_name = item
            self.WELL_NAMES.append(well_name)
            self.WELL_PATH[well_name] = item_path
            
            # 创建 DATA_WELL 对象 (懒加载，不立即扫描文件)
            from src_well_data.data_logging_well import DATA_WELL
            self.WELL_DATA[well_name] = DATA_WELL(
                path_folder=item_path,
                WELL_NAME=well_name
            )
    
    print(f"项目初始化完成：{len(self.WELL_NAMES)} 口井")
    print(f"井列表：{self.WELL_NAMES}")
```

### 3.2 获取单井数据 (get_well_data)

**目的：** 获取指定井的测井数据

**代码示例：**
```python
def get_well_data(self, well_name: str, file_path: str = '',
                  curve_names: List[str] = None, Norm: bool = False):
    """
    获取单井测井数据
    
    参数:
        well_name: 井名
        file_path: 文件路径关键字，为空时使用默认文件
        curve_names: 曲线名称列表
        Norm: 是否归一化
    
    返回:
        测井数据 DataFrame
    """
    if well_name not in self.WELL_DATA:
        raise ValueError(f"井 '{well_name}' 不在项目中")
    
    well = self.WELL_DATA[well_name]
    
    # 获取测井数据
    data = well.get_logging(
        key=file_path,
        curve_names=curve_names,
        norm=Norm
    )
    
    # 添加井名列
    if data is not None:
        data['Well'] = well_name
    
    return data
```

### 3.3 按特征获取数据 (get_well_data_by_charters)

**目的：** 根据路径特征关键字搜索并获取数据

**应用场景：** 当一口井有多个测井文件时，按特征选择特定文件

**代码示例：**
```python
def get_well_data_by_charters(self, well_name: str,
                               target_path_feature: List[str],
                               target_file_type: str,
                               curve_names: List[str] = None,
                               Norm: bool = False):
    """
    按特征获取单井数据
    
    参数:
        well_name: 井名
        target_path_feature: 路径特征关键字列表，如 ['Texture_ALL', '_80_5']
        target_file_type: 文件类型 ('logging', 'table', 'fmi', 'nmr')
        curve_names: 曲线名称列表
        Norm: 是否归一化
    
    返回:
        测井数据 DataFrame
    """
    # 先搜索文件路径
    file_paths = self.search_target_file_path(
        well_name=well_name,
        target_path_feature=target_path_feature,
        target_file_type=target_file_type
    )
    
    if not file_paths:
        raise ValueError(f"未找到匹配的文件：{target_path_feature}")
    
    # 使用第一个匹配的文件
    well = self.WELL_DATA[well_name]
    
    # 根据文件类型获取数据
    if target_file_type == 'logging':
        data = well.get_logging(key=file_paths[0], curve_names=curve_names, norm=Norm)
    elif target_file_type == 'table':
        data = well.get_table(key=file_paths[0])
    elif target_file_type == 'fmi':
        depth, image = well.get_FMI(key=file_paths[0])
        data = pd.DataFrame({'DEPTH': depth})
    elif target_file_type == 'nmr':
        depth, t2, spectra = well.get_NMR(key=file_paths[0])
        data = pd.DataFrame({'DEPTH': depth})
    else:
        raise ValueError(f"不支持的文件类型：{target_file_type}")
    
    if data is not None:
        data['Well'] = well_name
    
    return data
```

### 3.4 搜索文件路径 (search_target_file_path)

**目的：** 在指定井中搜索包含特定关键字的文件

**代码示例：**
```python
def search_target_file_path(self, well_name: str,
                             target_path_feature: List[str],
                             target_file_type: str):
    """
    搜索目标文件路径
    
    参数:
        well_name: 井名
        target_path_feature: 路径特征关键字列表
        target_file_type: 文件类型
    
    返回:
        匹配的文件路径列表
    """
    if well_name not in self.WELL_DATA:
        raise ValueError(f"井 '{well_name}' 不在项目中")
    
    well = self.WELL_DATA[well_name]
    
    # 根据文件类型选择搜索方法
    if target_file_type == 'logging':
        return well.search_logging_path_list(new_kw=target_path_feature)
    elif target_file_type == 'table':
        return well.search_table_path_list(new_kw=target_path_feature)
    elif target_file_type == 'fmi':
        return well.search_fmi_path_list(new_kw=target_path_feature)
    elif target_file_type == 'nmr':
        return well.search_nmr_path_list(new_kw=target_path_feature)
    else:
        raise ValueError(f"不支持的文件类型：{target_file_type}")
```

### 3.5 合并测井与表格数据 (combined_all_logging_with_type)

**目的：** 批量合并多井的测井数据与表格数据 (如岩性)

**原理：**
1. 遍历每口井
2. 获取测井数据和表格数据
3. 在单井内合并测井与表格
4. 添加井名列
5. 垂直拼接所有井的数据

**代码示例：**
```python
def combined_all_logging_with_type(self,
                                    well_names: List[str],
                                    file_path_logging: str,
                                    file_path_table: str,
                                    curve_names_logging: List[str],
                                    curve_names_type: List[str],
                                    replace_dict: Dict,
                                    type_new_col: str,
                                    Norm: bool = False):
    """
    合并测井数据与表格数据 (多井)
    
    参数:
        well_names: 井名列表
        file_path_logging: 测井文件路径关键字
        file_path_table: 表格文件路径关键字
        curve_names_logging: 测井曲线名称列表
        curve_names_type: 表格列名列表
        replace_dict: 类别替换字典
        type_new_col: 新列名
        Norm: 是否归一化
    
    返回:
        合并后的 DataFrame (包含 Well 列)
    """
    all_data = []
    
    for well_name in well_names:
        if well_name not in self.WELL_DATA:
            print(f"警告：井 '{well_name}' 不在项目中，跳过")
            continue
        
        well = self.WELL_DATA[well_name]
        
        # 合并单井的测井与表格数据
        combined = well.combine_logging_table(
            logging_key=file_path_logging,
            curve_names_logging=curve_names_logging,
            table_key=file_path_table,
            replace_dict=replace_dict,
            new_col=type_new_col,
            norm=Norm
        )
        
        if combined is not None:
            combined['Well'] = well_name
            all_data.append(combined)
    
    if not all_data:
        return None
    
    # 垂直拼接所有井的数据
    result = pd.concat(all_data, ignore_index=True)
    
    # 缓存
    self.data_vc = result
    
    return result
```

### 3.6 批量获取 FMI 纹理 (get_fmi_texture)

**目的：** 批量计算多井的 FMI 纹理特征

**代码示例：**
```python
def get_fmi_texture(self, well_names: List[str],
                    file_path_logging: str,
                    Mode: str = 'MEAN',
                    texture_config: Dict = None):
    """
    批量获取多井 FMI 纹理数据
    
    参数:
        well_names: 井名列表
        file_path_logging: 测井文件路径关键字
        Mode: 纹理模式 ('MEAN', 'SUB', 'X', 'Y')
        texture_config: 纹理计算配置
    
    返回:
        纹理数据 DataFrame (包含 Well 列)
    """
    import numpy as np
    
    if texture_config is None:
        texture_config = {
            'level': 16,
            'distance': [2, 4],
            'angles': [0, np.pi/2],
            'windows_length': 80,
            'windows_step': 10
        }
    
    all_texture = []
    
    for well_name in well_names:
        if well_name not in self.WELL_DATA:
            continue
        
        well = self.WELL_DATA[well_name]
        
        # 搜索 FMI 文件
        fmi_files = well.search_fmi_path_list(new_kw=['DYNA'])
        
        if not fmi_files:
            print(f"警告：井 '{well_name}' 没有 FMI 数据")
            continue
        
        # 获取纹理数据
        texture_df = well.get_FMI_texture(
            key=fmi_files[0],
            texture_config=texture_config
        )
        
        if texture_df is not None:
            # 筛选指定模式的列
            if Mode:
                mode_cols = [col for col in texture_df.columns if Mode in col.upper()]
                mode_cols = ['DEPTH'] + mode_cols
                texture_df = texture_df[mode_cols]
            
            texture_df['Well'] = well_name
            all_texture.append(texture_df)
    
    if not all_texture:
        return None
    
    return pd.concat(all_texture, ignore_index=True)
```

### 3.7 垂直链接多井数据 (vertical_concat_wells)

**目的：** 将多井数据垂直拼接成一个 DataFrame

**代码示例：**
```python
def vertical_concat_wells(self, well_names: List[str], **kwargs):
    """
    垂直链接多井数据
    
    参数:
        well_names: 井名列表
        **kwargs: 传递给 get_well_data 的参数
    
    返回:
        垂直合并后的 DataFrame
    """
    all_data = []
    
    for well_name in well_names:
        data = self.get_well_data(well_name=well_name, **kwargs)
        
        if data is not None:
            all_data.append(data)
    
    if not all_data:
        return None
    
    result = pd.concat(all_data, ignore_index=True)
    self.data_vc = result
    
    return result
```

---

## 四、使用示例

### 4.1 创建项目

```python
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT

# 创建项目
project = LOGGING_PROJECT(
    project_path=r'C:\logging_data\城 96 区块'
)

# 查看项目信息
print(f"项目路径：{project.PROJECT_PATH}")
print(f"井列表：{project.WELL_NAMES}")
print(f"井路径映射：{project.WELL_PATH}")
```

### 4.2 获取单井数据

```python
# 获取单井测井数据
data = project.get_well_data(
    well_name='城 96',
    curve_names=['GR', 'AC', 'DEN']
)
print(data.head())
# 输出包含 'Well' 列
```

### 4.3 按特征获取数据

```python
# 获取包含特定特征的测井文件数据
data = project.get_well_data_by_charters(
    well_name='城 96',
    target_path_feature=['Texture_ALL', '_80_5'],
    target_file_type='logging',
    curve_names=['GR', 'AC']
)
```

### 4.4 搜索文件路径

```python
# 搜索特定文件
fmi_files = project.search_target_file_path(
    well_name='城 96',
    target_path_feature=['DYNA'],
    target_file_type='fmi'
)
print(f"FMI 文件：{fmi_files}")

# 搜索纹理测井文件
texture_files = project.search_target_file_path(
    well_name='城 96',
    target_path_feature=['120', 'TEXTURE'],
    target_file_type='logging'
)
print(f"纹理文件：{texture_files}")
```

### 4.5 合并测井与岩性数据 (多井)

```python
# 合并多井测井与岩性数据
data_combined = project.combined_all_logging_with_type(
    well_names=['城 96', '元 543', '云安 012'],
    file_path_logging='logging_data',
    file_path_table='LITHO_TYPE',
    curve_names_logging=['GR', 'AC', 'DEN'],
    curve_names_type=['Type'],
    replace_dict={'砂岩': 0, '泥岩': 1, '砂砾岩': 2},
    type_new_col='Lithology'
)

print(data_combined.columns.tolist())
# ['DEPTH', 'GR', 'AC', 'DEN', 'Lithology', 'Well']

print(data_combined['Well'].unique())
# ['城 96', '元 543', '云安 012']
```

### 4.6 批量计算 FMI 纹理

```python
import numpy as np

# 配置纹理参数
texture_config = {
    'level': 16,
    'distance': [2, 4],
    'angles': [0, np.pi/2],
    'windows_length': 80,
    'windows_step': 10
}

# 批量获取多井 FMI 纹理
texture_data = project.get_fmi_texture(
    well_names=['城 96', '元 543'],
    file_path_logging='texture_logging_120',
    Mode='MEAN',
    texture_config=texture_config
)

print(texture_data.columns.tolist())
# ['DEPTH', 'CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', ..., 'Well']
```

### 4.7 垂直链接多井数据

```python
# 垂直链接多井测井数据
data_vc = project.vertical_concat_wells(
    well_names=['城 96', '元 543', '云安 012'],
    curve_names=['GR', 'AC', 'DEN']
)

print(f"总数据量：{len(data_vc)}")
print(f"深度范围：{data_vc['DEPTH'].min():.2f} - {data_vc['DEPTH'].max():.2f} m")
```

### 4.8 访问缓存的垂直合并数据

```python
# 访问之前缓存的垂直合并数据
cached_data = project.data_vc

# 按井筛选
well_96_data = cached_data[cached_data['Well'] == '城 96']
```

---

## 五、项目结构组织

### 5.1 推荐的项目目录结构

```
项目根目录/
├── 城 96/
│   ├── 城 96_logging_data.csv
│   ├── 城 96__LITHO_TYPE.csv
│   ├── 城 96-DYNA.txt
│   ├── 城 96-STAT.txt
│   └── 城 96_nmr_data.csv
│
├── 元 543/
│   ├── 元 543_logging_data.csv
│   ├── 元 543__LITHO_TYPE.csv
│   ├── 元 543-DYNA.txt
│   └── 元 543-STAT.txt
│
└── 云安 012/
    ├── 云安 012_logging_data.csv
    └── 云安 012__LITHO_TYPE.csv
```

### 5.2 项目初始化流程

```
1. 指定项目根路径
   ↓
2. 扫描一级子文件夹 (每口井)
   ↓
3. 建立井名 - 路径映射
   ↓
4. 为每口井创建 DATA_WELL 对象
   ↓
5. 项目初始化完成
```

---

## 六、与单井模块的对比

| 特性 | LOGGING_PROJECT | DATA_WELL |
|------|-----------------|-----------|
| **管理范围** | 多井 | 单井 |
| **数据获取** | 需指定井名 | 直接获取 |
| **返回数据** | 含 Well 列 | 不含 Well 列 |
| **文件搜索** | 指定井内搜索 | 全文件夹搜索 |
| **数据合并** | 支持多井垂直合并 | 支持测井 - 表格合并 |
| **典型用途** | 多井对比、区域分析 | 单井分析、可视化 |

---

## 七、性能优化

### 7.1 懒加载继承

- LOGGING_PROJECT 不直接加载数据
- 数据加载委托给各 DATA_WELL 对象
- DATA_WELL 采用懒加载策略
- 整体实现按需加载

### 7.2 批量处理优化

```python
# 优化：预加载常用数据
def preload_wells(self, well_names: List[str], **kwargs):
    """预加载指定井的数据"""
    for well_name in well_names:
        if well_name in self.WELL_DATA:
            well = self.WELL_DATA[well_name]
            well.get_logging(**kwargs)  # 触发加载
```

### 7.3 内存管理

```python
def cleanup_project(self, well_names: List[str] = None):
    """
    清理项目数据，释放内存
    
    参数:
        well_names: 要清理的井名列表，None 表示全部
    """
    wells_to_clean = well_names if well_names else self.WELL_NAMES
    
    for well_name in wells_to_clean:
        if well_name in self.WELL_DATA:
            well = self.WELL_DATA[well_name]
            well.cleanup()  # 调用单井的清理方法
    
    # 清理垂直合并缓存
    self.data_vc = None
    
    import gc
    gc.collect()
```

---

## 八、常见问题

### Q1: 井名识别不正确？

**可能原因：**
- 子文件夹命名包含非井名字符
- 存在非井数据文件夹

**解决方案：**
```python
# 手动指定井列表
project.WELL_NAMES = ['城 96', '元 543', '云安 012']
project.WELL_PATH = {
    '城 96': r'C:\data\城 96',
    '元 543': r'C:\data\元 543',
    '云安 012': r'C:\data\云安 012'
}

# 重新创建 DATA_WELL 对象
for well_name in project.WELL_NAMES:
    project.WELL_DATA[well_name] = DATA_WELL(
        path_folder=project.WELL_PATH[well_name],
        WELL_NAME=well_name
    )
```

### Q2: 多井数据深度范围不一致？

**解决方案：**
```python
# 检查各井深度范围
for well_name in project.WELL_NAMES:
    data = project.get_well_data(well_name, curve_names=['GR'])
    print(f"{well_name}: {data['DEPTH'].min():.2f} - {data['DEPTH'].max():.2f} m")

# 统一深度范围
common_depth_min = max(project.get_well_data(w, ['GR'])['DEPTH'].min() 
                       for w in project.WELL_NAMES)
common_depth_max = min(project.get_well_data(w, ['GR'])['DEPTH'].max() 
                       for w in project.WELL_NAMES)

# 使用统一深度范围获取数据
data = project.get_well_data(
    well_name='城 96',
    curve_names=['GR', 'AC'],
    depth_limit=[common_depth_min, common_depth_max]
)
```

### Q3: 垂直合并后数据混乱？

**可能原因：**
- 各井曲线列名不一致
- 数据类型不一致

**解决方案：**
```python
# 确保各井使用相同的曲线列表
curve_names = ['GR', 'AC', 'DEN', 'CNL']

data_vc = project.vertical_concat_wells(
    well_names=['城 96', '元 543'],
    curve_names=curve_names
)

# 检查合并后的数据
print(data_vc.info())
print(data_vc['Well'].value_counts())
```

---

## 九、扩展方向

### 9.1 可能的功能扩展

1. **项目配置保存：** 保存和加载项目配置
2. **井分组管理：** 支持井的分组 (如按构造、按层系)
3. **数据导出：** 批量导出多井数据
4. **统计分析：** 多井数据统计分析功能
5. **对比分析：** 多井曲线对比、地层对比

### 9.2 与可视化层集成

```python
# 为多井可视化准备数据
def prepare_multi_well_viz_data(project, well_names, curve_names):
    """准备多井可视化数据"""
    viz_data = {}
    
    for well_name in well_names:
        data = project.get_well_data(
            well_name=well_name,
            curve_names=curve_names
        )
        viz_data[well_name] = data
    
    return viz_data

# 使用
viz_data = prepare_multi_well_viz_data(
    project,
    well_names=['城 96', '元 543'],
    curve_names=['GR', 'AC', 'DEN']
)
```

---

## 十、最佳实践

### 10.1 项目组织建议

1. **井文件夹命名：** 使用统一的井名格式
2. **文件命名规范：** 遵循模块 05 中的命名规范
3. **项目根路径：** 避免过深的路径嵌套
4. **数据备份：** 定期备份项目配置

### 10.2 使用模式

```python
# 推荐的使用模式

# 1. 创建项目
project = LOGGING_PROJECT(project_path=r'C:\logging_data\区块 A')

# 2. 查看项目包含的井
print(f"项目包含 {len(project.WELL_NAMES)} 口井")
print(project.WELL_NAMES)

# 3. 获取单井数据
single_well_data = project.get_well_data(
    well_name='城 96',
    curve_names=['GR', 'AC', 'DEN']
)

# 4. 多井合并分析
combined_data = project.combined_all_logging_with_type(
    well_names=project.WELL_NAMES,
    file_path_logging='logging_data',
    file_path_table='LITHO_TYPE',
    curve_names_logging=['GR', 'AC'],
    curve_names_type=['Type'],
    replace_dict={'砂岩': 0, '泥岩': 1},
    type_new_col='Lithology'
)

# 5. 按需清理内存
project.cleanup_project(well_names=['城 96'])
```

---

*最后更新：2026-03-14*  
*Well Logging Visualization System - Well Project Module v1.0*
