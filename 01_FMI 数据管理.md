# 📡 FMI 数据管理模块

> FMI (Formation MicroScanner Image) Data Management Module

**版本：** v1.0  
**最后更新：** 2026-03-14  
**关联源文件：** `src_well_data/data_logging_FMI.py`, `src_fmi/`

---

## 一、模块概述

### 1.1 模块定位

FMI 数据管理模块专门负责**电成像测井数据**的加载、处理和纹理特征计算。电成像测井通过多个极板测量井壁电阻率，生成高分辨率的井壁图像，用于识别地质构造、裂缝、层理等特征。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| **数据加载** | 读取 FMI 动态 (DYNA) 和静态 (STAT) 数据文件 |
| **数据解析** | 解析 TXT 格式的极板/按钮电阻率数据 |
| **纹理计算** | 基于灰度共生矩阵 (GLCM) 计算纹理特征 |
| **数据封装** | 统一的数据访问接口 |

### 1.3 技术特点

- 支持动态 (DYNA) 和静态 (STAT) 两种 FMI 数据格式
- 自动跳过文件头 (默认前 8 行)
- 支持多极板/多按钮数据并行处理
- 纹理特征计算可配置窗口大小和步长

---

## 二、文件结构与格式

### 2.1 源文件组织

```
src_fmi/
├── fmi_data_read.py           # FMI 数据读取核心
├── fmi_fractal_dimension.py   # 分形维数计算
├── image_operation.py         # 图像处理操作
└── texture_analysis.py        # 纹理特征分析 (隐含)

src_well_data/
└── data_logging_FMI.py        # DataFMI 类定义
```

### 2.2 数据文件格式

#### FMI 原始数据 (TXT)

**命名规则：** `{井名}-DYNA.txt` 或 `{井名}-STAT.txt`

**文件结构：**
```
# 前 8 行为文件头 (元数据、测井信息等)
# 第 9 行开始为数据行
深度值，极板 1_按钮 1, 极板 1_按钮 2, ..., 极板 N_按钮 M
3200.5, 125.3, 128.7, ..., 142.1
3200.6, 126.1, 129.2, ..., 143.0
...
```

**数据特点：**
- 每行第一个数据为深度值 (单位：米)
- 后续列为各极板/按钮的电阻率或电导率测量值
- 默认跳过前 8 行文件头

#### 纹理数据 (CSV)

**命名规则：** `{井名}_texture_logging_{窗长}.csv`

**文件结构：**
| 列名 | 类型 | 说明 |
|------|------|------|
| DEPTH | float | 深度 (m) |
| CON_MEAN_STAT | float | 对比度均值 (静态) |
| DIS_MEAN_STAT | float | 差异性均值 (静态) |
| HOM_MEAN_STAT | float | 均匀度均值 (静态) |
| ENG_MEAN_STAT | float | 能量均值 (静态) |
| COR_MEAN_STAT | float | 相关性均值 (静态) |
| CON_MEAN_DYNA | float | 对比度均值 (动态) |
| ... | ... | 其他纹理特征 |

---

## 三、DataFMI 类设计

### 3.1 类结构

```python
class DataFMI:
    """FMI 数据管理类"""
    
    def __init__(self, path: str, well_name: str):
        """
        初始化 FMI 数据对象
        
        参数:
            path: FMI 数据文件路径
            well_name: 井名
        """
        self.path = path
        self.well_name = well_name
        self.data = None
        self.depth = None
        self.image_data = None
    
    def load_data(self) -> None:
        """加载 FMI 数据文件"""
        pass
    
    def get_data(self, depth: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取 FMI 数据
        
        参数:
            depth: 可选的深度范围限制
        
        返回:
            (depth_array, image_data) 元组
        """
        pass
    
    def calculate_texture(self, texture_config: Dict) -> pd.DataFrame:
        """
        计算纹理特征
        
        参数:
            texture_config: 纹理计算配置
                - level: 灰度级数 (默认 16)
                - distance: 距离列表 [2, 4]
                - angles: 角度列表 [0, π/2]
                - windows_length: 窗口长度 (默认 80)
                - windows_step: 窗口步长 (默认 10)
        
        返回:
            纹理特征 DataFrame
        """
        pass
```

### 3.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `path` | str | FMI 数据文件路径 |
| `well_name` | str | 所属井名 |
| `data` | np.ndarray | 原始数据数组 |
| `depth` | np.ndarray | 深度数组 |
| `image_data` | np.ndarray | 图像数据 (极板×按钮) |
| `texture_data` | pd.DataFrame | 纹理特征数据 (缓存) |

### 3.3 核心方法详解

#### load_data()

**目的：** 从 TXT 文件加载 FMI 原始数据

**原理：**
1. 使用 `np.loadtxt()` 读取文件
2. 跳过前 8 行文件头 (`skiprows=8`)
3. 第一列作为深度，其余列作为图像数据
4. 数据存储在 `self.depth` 和 `self.image_data`

**代码示例：**
```python
def load_data(self):
    raw_data = np.loadtxt(self.path, skiprows=8)
    self.depth = raw_data[:, 0]
    self.image_data = raw_data[:, 1:]
    self.data = raw_data
```

#### get_data(depth)

**目的：** 获取指定深度范围的 FMI 数据

**参数：**
- `depth`: 可选，深度范围 `[depth_min, depth_max]`

**返回：**
- `depth_array`: 深度数组
- `image_data`: 对应的图像数据

**原理：**
1. 如果未指定 depth，返回全部数据
2. 如果指定 depth，进行深度范围筛选
3. 返回筛选后的数据

#### calculate_texture(texture_config)

**目的：** 基于灰度共生矩阵 (GLCM) 计算纹理特征

**纹理特征说明：**

| 特征 | 前缀 | 物理意义 |
|------|------|----------|
| Contrast | CON | 对比度，反映图像局部变化程度 |
| Dissimilarity | DIS | 差异性，衡量像素差异 |
| Homogeneity | HOM | 均匀度，反映图像均匀程度 |
| Energy | ENG | 能量，反映纹理粗细程度 |
| Correlation | COR | 相关性，衡量像素线性相关性 |
| Entropy | ENT | 熵，反映图像复杂度 |
| ASM | ASM | 角二阶矩，反映纹理均匀性 |

**计算原理：**
1. **灰度量化：** 将电阻率值量化为指定灰度级 (默认 16 级)
2. **滑动窗口：** 沿深度方向滑动窗口 (窗长 80，步长 10)
3. **GLCM 计算：** 对每个窗口计算灰度共生矩阵
4. **特征提取：** 从 GLCM 提取 7 种纹理特征
5. **方向平均：** 对多个方向 (0°, 90°) 的结果取平均
6. **距离平均：** 对多个距离 (2, 4) 的结果取平均

**配置参数：**
```python
texture_config = {
    'level': 16,              # 灰度级数
    'distance': [2, 4],       # GLCM 距离列表
    'angles': [0, np.pi/2],   # GLCM 角度列表
    'windows_length': 80,     # 滑动窗口长度 (深度点数)
    'windows_step': 10        # 滑动步长
}
```

---

## 四、在 DATA_WELL 中的集成

### 4.1 文件识别

```python
class DATA_WELL:
    # FMI 文件识别关键字
    FMI_KW = ['DYNA', 'STAT']
    
    def scan_files(self):
        """扫描井文件夹中的 FMI 文件"""
        self.path_list_fmi = search_files_by_criteria(
            self.well_path,
            name_keywords=self.FMI_KW,
            file_extensions=['.txt'],
            all_keywords=False
        )
```

### 4.2 数据初始化

```python
def init_FMI(self, path: str = ''):
    """
    初始化 FMI 数据对象 (懒加载)
    
    参数:
        path: FMI 文件路径，为空时使用默认路径
    """
    if path not in self.FMI_dict:
        target_path = path if path else self.path_list_fmi[0]
        self.FMI_dict[target_path] = DataFMI(
            path=target_path,
            well_name=self.WELL_NAME
        )
```

### 4.3 数据访问接口

```python
def get_FMI(self, key: str = '', depth: Optional[List[float]] = None):
    """
    获取 FMI 数据
    
    参数:
        key: FMI 文件路径关键字
        depth: 深度范围限制
    
    返回:
        (depth_array, image_data) 元组
    """
    self.init_FMI(key)
    obj = self._get_default_obj(self.FMI_dict, key)
    return obj.get_data(depth) if obj else None
```

### 4.4 纹理数据获取

```python
def get_FMI_texture(self, key: str = '', texture_config: Dict = None):
    """
    获取 FMI 纹理特征数据
    
    参数:
        key: FMI 文件路径关键字
        texture_config: 纹理计算配置
    
    返回:
        纹理特征 DataFrame
    """
    self.init_FMI(key)
    obj = self._get_default_obj(self.FMI_dict, key)
    if obj and texture_config:
        return obj.calculate_texture(texture_config)
    return None
```

### 4.5 文件搜索辅助方法

```python
def search_fmi_path_list(self, new_kw: List[str]) -> List[str]:
    """
    搜索特定 FMI 文件路径
    
    参数:
        new_kw: 额外关键字列表，如 ['DYNA'] 或 ['STAT']
    
    返回:
        匹配的文件路径列表
    """
    return self._search_path_list(
        base_list=self.path_list_fmi,
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

# 获取 FMI 文件列表
fmi_files = well.get_path_list_fmi()
print(f"找到 {len(fmi_files)} 个 FMI 文件")

# 搜索动态和静态文件
fmi_dyna = well.search_fmi_path_list(new_kw=['DYNA'])
fmi_stat = well.search_fmi_path_list(new_kw=['STAT'])
```

### 5.2 获取 FMI 图像数据

```python
# 获取 FMI 动态数据
depth, image_data = well.get_FMI(key=fmi_dyna[0])

print(f"深度范围：{depth[0]:.2f} - {depth[-1]:.2f} m")
print(f"图像数据形状：{image_data.shape}")  # (深度点数，极板数×按钮数)
```

### 5.3 计算纹理特征

```python
import numpy as np

# 配置纹理计算参数
texture_config = {
    'level': 16,
    'distance': [2, 4],
    'angles': [0, np.pi/2],
    'windows_length': 80,
    'windows_step': 10
}

# 计算并获取纹理数据
texture_df = well.get_FMI_texture(
    key=fmi_dyna[0],
    texture_config=texture_config
)

print(texture_df.columns.tolist())
# ['DEPTH', 'CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', ...]
```

### 5.4 在 PROJECT 层使用

```python
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT

# 创建项目
project = LOGGING_PROJECT(project_path=r'C:\logging_data')

# 获取多井 FMI 纹理数据
texture_data = project.get_fmi_texture(
    well_names=['城 96', '元 543'],
    file_path_logging='texture_logging_120',
    Mode='MEAN',
    texture_config=texture_config
)
```

---

## 六、可视化集成

### 6.1 可视化配置

```python
config_fmi = {
    'color_map': 'rainbow',      # 颜色映射
    'scale_range': [0, 256],     # 值范围
    'title_fmi': ['FMI 动态', 'FMI 静态'],  # 标题
    'show_title': True           # 显示标题
}
```

### 6.2 数据格式转换

```python
# 为可视化准备 FMI 数据字典
fmi_dict = {
    'depth': depth_array,
    'image_data': [fmi_dyna_data, fmi_stat_data],
    'title': ['FMI 动态', 'FMI 静态']
}
```

### 6.3 渲染原理

1. **数据准备：** 将 FMI 图像数据转换为 matplotlib 可接受的格式
2. **热力图渲染：** 使用 `imshow()` 绘制电阻率热力图
3. **深度对齐：** Y 轴与测井曲线共享深度坐标
4. **颜色映射：** 应用彩虹色或其他颜色映射显示电阻率变化

---

## 七、性能优化

### 7.1 懒加载策略

- FMI 数据文件较大，采用懒加载策略
- 仅在首次调用 `get_FMI()` 时加载数据
- 后续访问直接使用缓存的 `self.data`

### 7.2 纹理计算优化

**优化点：**
1. **窗口滑动：** 使用步长跳过部分深度点，减少计算量
2. **灰度量化：** 降低灰度级数可加速 GLCM 计算
3. **缓存机制：** 计算后的纹理数据可缓存复用

**配置建议：**
```python
# 快速预览 (低精度)
texture_config_fast = {
    'level': 8,
    'distance': [2],
    'angles': [0],
    'windows_length': 40,
    'windows_step': 20
}

# 精细分析 (高精度)
texture_config_fine = {
    'level': 32,
    'distance': [2, 4, 6],
    'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'windows_length': 120,
    'windows_step': 5
}
```

---

## 八、常见问题

### Q1: FMI 文件读取失败？

**可能原因：**
- 文件格式不是纯 TXT (可能包含特殊字符)
- 文件头行数不是 8 行
- 数据列数不一致

**解决方案：**
```python
# 手动检查文件
with open(fmi_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 15:  # 查看前 15 行
            print(f"Line {i}: {line[:100]}")
```

### Q2: 纹理计算速度太慢？

**优化方案：**
1. 增大 `windows_step` (如从 10 改为 20)
2. 减少 `distance` 和 `angles` 的数量
3. 降低 `level` 灰度级数
4. 缩短 `windows_length`

### Q3: 如何区分 DYNA 和 STAT？

- **DYNA (动态)：** 电阻率值经过动态范围调整，适合观察局部变化
- **STAT (静态)：** 电阻率值使用固定范围，适合整体对比

建议同时加载两种数据，对比分析。

---

## 九、扩展方向

### 9.1 可能的功能扩展

1. **分形维数计算：** 已部分实现于 `fmi_fractal_dimension.py`
2. **裂缝识别：** 基于纹理特征的自动裂缝检测
3. **层理分析：** 倾角和倾向计算
4. **图像增强：** 直方图均衡化、滤波去噪

### 9.2 与其他模块的协同

- **与常规测井协同：** FMI 纹理 + 常规曲线联合解释
- **与表格数据协同：** FMI 特征 + 岩性标签训练分类模型
- **与 NMR 协同：** 孔隙结构多尺度表征

---

*最后更新：2026-03-14*  
*Well Logging Visualization System - FMI Module v1.0*
