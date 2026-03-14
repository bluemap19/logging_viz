# 📊 测井数据可视化系统 (Well Logging Visualization System)

> 专业级多类型测井数据统一管理与交互式可视化平台

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Development-yellow.svg)

---

## 📖 项目简介

本系统是一个专业的测井数据可视化平台，支持**常规测井、电成像 (FMI)、核磁共振 (NMR)、离散表格数据**的统一管理与交互式可视化展示。

### 核心特性

| 特性 | 说明 |
|------|------|
| **多数据类型** | 常规测井、电成像 (FMI)、离散表格、核磁 (NMR) |
| **文件自动识别** | 基于关键字自动扫描和分类文件 |
| **统一数据管理** | 单井数据统一封装，多井项目统一管理 |
| **交互式可视化** | 滑动条控制、深度导航、自适应布局 |
| **高性能缓存** | 深度窗口数据缓存，支持快速滚动刷新 |
| **可扩展架构** | 模块化设计，易于添加新数据类型 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────┐
│         项目层 (LOGGING_PROJECT)         │
│         多井项目管理与数据整合            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         单井层 (DATA_WELL)               │
│         单井数据统一管理与访问            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      数据类型层 (Data Type Layer)        │
│  DataLogging │ DataFMI │ DataTable │ DataNMR │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      可视化层 (Visualization Layer)      │
│   WellLogVisualizer │ LoggingDataManager │
└─────────────────────────────────────────┘
```

---

## 📁 项目结构

```
logging_viz/
├── src_well_project/          # 项目管理层
│   └── LOGGING_PROJECT.py     # 多井项目管理类
│
├── src_well_data/             # 单井数据层
│   ├── data_logging_well.py   # 单井数据统一管理
│   ├── data_logging_normal.py # 常规测井数据
│   ├── data_logging_FMI.py    # 电成像数据
│   ├── data_logging_table.py  # 表格数据
│   └── data_logging_NMR.py    # 核磁数据
│
├── src_plot/                  # 可视化层
│   ├── data_visulization.py   # 可视化主类
│   ├── data_manager.py        # 数据管理器
│   └── cache_logging.py       # 缓存管理
│
├── src_file_op/               # 文件操作层
│   ├── dir_operation.py       # 目录扫描与文件搜索
│   └── xlsx_file_read.py      # Excel 文件读取
│
├── src_data_process/          # 数据处理层
│   ├── data_correction_analysis.py
│   ├── data_depth_delete.py
│   ├── data_filter.py
│   ├── data_norm.py
│   └── ...
│
├── src_fmi/                   # 电成像专用处理
│   ├── fmi_data_read.py
│   ├── fmi_fractal_dimension.py
│   └── image_operation.py
│
├── src_table/                 # 表格数据处理
│   └── table_process.py
│
├── docs/                      # 开发文档
│   ├── 开发文档.md
│   ├── 01_FMI 数据管理.md
│   ├── 02_NMR 数据管理.md
│   ├── 03_常规测井数据管理.md
│   ├── 04_表格数据管理.md
│   ├── 05_单井数据管理.md
│   ├── 06_井群 Project 数据管理.md
│   └── 07_单井测井数据可视化及可视化管理.md
│
├── tests/                     # 测试用例
│   └── test_visualization.py
│
├── examples/                  # 使用示例
│   └── example_single_well.py
│
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目说明
└── .gitignore                 # Git 忽略文件
```

---

## 🚀 快速开始

### 3.1 环境要求

- Python 3.8+
- Windows / Linux / macOS

### 3.2 安装依赖

```bash
# 克隆项目
git clone https://github.com/你的用户名/logging_viz.git
cd logging_viz

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 3.3 基础使用示例

#### 单井可视化

```python
from src_well_data.data_logging_well import DATA_WELL
from src_plot.data_manager import LoggingDataManager
from src_plot.data_visulization import WellLogVisualizer

# 1. 加载单井数据
well = DATA_WELL(
    path_folder=r'F:\logging_workspace\井名',
    WELL_NAME='井名'
)

# 2. 准备数据
logging_data = well.combine_logging_table(
    logging_key=well.get_path_list_logging()[0],
    table_key=well.get_path_list_table()[0],
    curve_names_logging=['GR', 'AC', 'DEN']
)

# 3. 创建数据管理器
LDM = LoggingDataManager(
    logging_data=logging_data,
    fmi_data={'depth': depth_fmi, 'image_data': [data_fmi]},
    nmr_data={'depth': depth_nmr, 'nmr_data': [data_nmr]}
)

# 4. 创建可视化器并显示
viewer = WellLogVisualizer(data_manager=LDM)
viewer.visualize(figsize=(16, 10))
```

#### 多井项目管理

```python
from src_well_project.LOGGING_PROJECT import LOGGING_PROJECT

# 创建项目
project = LOGGING_PROJECT(project_path=r'C:\logging_data\区块名')

# 获取单井数据
data = project.get_well_data(well_name='井 1', curve_names=['GR', 'AC'])

# 合并多井数据
data_combined = project.combined_all_logging_with_type(
    well_names=['井 1', '井 2', '井 3'],
    file_path_logging='logging_data',
    file_path_table='LITHO_TYPE',
    curve_names_logging=['GR', 'AC', 'DEN'],
    type_new_col='Lithology'
)
```

---

## 📚 文档说明

详细的开发文档位于 `docs/` 目录：

| 文档 | 内容 |
|------|------|
| `01_FMI 数据管理.md` | FMI 数据加载、纹理计算、GLCM 特征 |
| `02_NMR 数据管理.md` | NMR T2 谱处理、孔隙度计算 |
| `03_常规测井数据管理.md` | 常规曲线处理、滤波、归一化 |
| `04_表格数据管理.md` | 岩性表格、类别编码、数据合并 |
| `05_单井数据管理.md` | DATA_WELL 类、文件扫描、懒加载 |
| `06_井群 Project 数据管理.md` | 多井项目、数据合并、垂直链接 |
| `07_单井测井数据可视化及可视化管理.md` | 可视化器、数据管理器、缓存系统 |

---

## 🎯 核心功能

### 数据类型支持

| 类型 | 文件格式 | 说明 |
|------|----------|------|
| 常规测井 | CSV/XLSX | GR, AC, DEN, CNL, RT 等曲线 |
| FMI 电成像 | TXT | DYNA/STAT 格式，支持纹理计算 |
| NMR 核磁 | CSV | T2 谱数据，支持对数坐标显示 |
| 表格数据 | CSV/XLSX | 岩性、沉积相等分类数据 |

### 可视化功能

- ✅ 多曲线道并排显示
- ✅ 岩性分类填充柱状图
- ✅ FMI 图像热力图显示
- ✅ NMR T2 谱图（支持对数坐标）
- ✅ 交互式深度导航（滑动条 + 滚轮）
- ✅ 图例自动/手动配置

---

## 🛠️ 开发指南

### 添加新数据类型

1. 在 `src_well_data/` 创建新的数据类
2. 在 `DATA_WELL` 中注册文件识别关键字
3. 在 `WellLogVisualizer` 中添加渲染方法
4. 在 `LoggingDataManager` 中添加数据验证

### 自定义可视化样式

```python
# 修改颜色配置
config_logging = {
    'curve_colors': ['#你的颜色 1', '#你的颜色 2', ...],
    'line_width': [1.0, 1.5, ...]
}

# 修改布局配置
LAYOUT_CONFIG = {
    'left_margin': 0.08,
    'title_height': 0.05,
    ...
}
```

---

## 📝 常见问题

### Q1: 中文显示为方块？
确保 matplotlib 字体配置正确：
```python
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
```

### Q2: 文件识别不到？
检查文件命名是否符合规范：
- 测井文件：`{井名}_logging_data.csv`
- FMI 文件：`{井名}-DYNA.txt` 或 `{井名}-STAT.txt`
- 岩性表格：`{井名}__LITHO_TYPE.csv`
- NMR 文件：包含 `nmr` 关键字

### Q3: 滚动时卡顿？
- 启用缓存系统
- 降低 FMI 像素密度配置
- 减少同时显示的数据类型

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 👥 作者与维护者

- **Author:** Fuhao Zhang
- **Email:** [你的邮箱]
- **Research Area:** Well Logging Engineering, Geological Engineering, Deep Learning, Image Processing

---

## 🙏 致谢

感谢所有为该项目做出贡献的开发者和使用者的反馈！

---

*Last Updated: 2026-03-14*
