import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.collections import PolyCollection
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import logging
from src_plot.visulization.data_manager import LoggingDataManager
from src_well_data_base.data_logging_well import DATA_WELL


class WellLogVisualizer:
    """
    测井数据可视化类 - 专业级可视化工具

    核心功能：
    - 多类型数据显示：常规测井曲线、岩性分类数据、FMI图像数据
    - 交互式导航：滑动条控制窗口大小，鼠标滚轮滚动深度
    - 自适应布局：根据数据类型自动调整面板布局
    """

    # ========== 颜色配置 ==========
    DEFAULT_TYPE_COLORS = ['#00FFFF', '#00FF00', '#FFFF00', '#FF4500', '#000000']

    # # 测井曲线道配置
    # DEFAULT_CURVE_COLORS = ['#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#8000FF', '#00FF80', '#FF0080', '#FFA500', '#FFFF00']
    # 测井曲线道配置
    DEFAULT_CURVE_COLORS = [
        '#FF0000',  # 红色 (保留)
        '#006400',  # 深绿色 (替换荧光绿色#00FF00)
        '#00008B',  # 深蓝色 (替换荧光蓝色#0000FF)
        '#00CED1',  # 深青色 (替换荧光青色#00FFFF)
        '#8B008B',  # 深洋红 (替换荧光洋红#FF00FF)
        '#4B0082',  # 靛蓝色 (替换#8000FF)
        '#006D5B',  # 深青绿 (替换#00FF80)
        '#8B0000',  # 深红色 (替换#FF0080)
        '#FF8C00',  # 深橙色 (替换#FFA500)
        '#B8860B'  # 暗金色 (替换#FFFF00)
    ]

    # 多曲线道颜色序列（用于同一道内的多条曲线）
    DEFAULT_MULTI_CURVE_COLORS = ['#FF0000', '#000000', '#0000FF', '#800080', '#FFA500', '#FFFF00']

    # 布局配置常量：定义图形各元素的相对位置和尺寸
    LAYOUT_CONFIG = {
        'left_margin': 0.06,  # 左侧边距占图形宽度的比例
        'right_margin': 0.98,  # 右侧边距（为滑动条留空间）
        'legend_bottom_margin': 0.067,  # 有图例时的底部边距
        'no_legend_bottom_margin': 0.062,  # 无图例时的底部边距
        'title_margin': 0.001,  # 标题框与子图的间距
        'title_height': 0.04,  # 标题框（数据头）高度，也决定了测井曲线图的图头高度
        'slider_width': 0.02,  # 滑动条宽度
        'min_window_size_ratio': 0.01,  # 最小窗口大小相对于总深度范围的比例
        'scroll_step_ratio': 0.1,  # 滚轮滚动步长相对于窗口大小的比例
        'axis_label_fontsize': 10,  # 坐标轴标签字体大小
        'tick_label_fontsize': 8,  # 刻度标签字体大小,右侧窗长滑动条字体大小
        'title_fontsize': 10,  # 标题字体大小

        'legend_box_alpha': 0.8,  # 图例框背景透明度
        'legend_facecolor': '#FFFFFF',  # 图例框颜色
        'legend_fontsize': 10,  # 图例字体大小

        'depth_indicator_alpha': 0.2,  # 深度指示器背景透明度
        'depth_indicator_facecolor':'#666666',# 深度指示器背景颜色
        'depth_indicator_fontsize':10,# 深度指示器字体大小
    }

    # ========== 渲染配置 ==========
    RENDERING_CONFIG = {
        'curve_linewidth': 1.0,  # 曲线线宽，常规测井的曲线线宽
        'title_box_linewidth': 1,  # 标题框边框宽度
        'head_box_facecolor': '#f5f5f5',  # 道头的背景颜色-浅灰色背景
        'head_box_edgecolor': '#aaaaaa',  # 道头的边框颜色-灰色边框
    }

    NMR_CONFIG = {
        'X_LOG': [True, True],           # 是否使用对数坐标轴
        'NMR_TITLE': ['N1', 'N2'],       # NMR面板标题
        'X_LIMIT': [[0.1, 1000], [0.1, 1000]],  # X轴显示范围
        'Y_scaling_factor': 1.0,         # Y轴缩放因子
        'JUMP_POINT': 50,                # 采样跳跃点数
        'line_width': 0.8,               # 谱线宽度
        'line_alpha': 0.9,               # 谱线透明度
        'fill_alpha': 0.5,               # 填充透明度
        'default_color': 'green',        # 默认颜色
        'fill_line_width': 0.8,          # 网格线透明度
        'amplitude_scale_base': 1.5,     # 振幅缩放基准值
        'amplitude_power_factor': 0.6    # 振幅幂次因子
    }

    FMI_CONFIG= {
        'MAP': '',
    }

    # ========== 字体配置 ==========
    FONT_CONFIG = {
        'family': ['DejaVu Sans', 'SimHei', 'sans-serif'],  # 字体族
        'weight_bold': 'bold',  # 粗体字重
    }

    def __init__(self,
        data_manager: LoggingDataManager=None,
        config_logging={},
        config_fmi: Dict[str, Any]={},
        config_nmr: Dict[str, Any]={},
        config_type: Dict[str, Any]={}) -> None:
        """
        初始化测井数据可视化器
        LoggingDataManager 测井数据管理器
        config_logging 常规测井绘图配置
        config_fmi：电成像绘图配置
        config_nmr： 核磁绘图配置
        config_type： 分类结果绘图配置
        """
        # 使用数据管理器检查各个配置项，看一下各个配置项是否合适
        validated_configs = data_manager.plot_config_check(config_logging=config_logging, config_fmi=config_fmi, config_nmr=config_nmr, config_type=config_type)

        self.data_manager = data_manager

        self.config_logging = validated_configs['logging']
        self.config_fmi = validated_configs['fmi']
        self.config_nmr = validated_configs['nmr']
        self.config_type = validated_configs['type']

        self.logger = data_manager.get_class_logger()

        self.logging_data_windows = pd.DataFrame
        self.fmi_data_windows = {}
        self.nmr_data_windows = {}
        self.type_data_windows = {}

        # 显示状态属性：记录当前视图的状态
        self.depth_min: float = 0.0  # 数据最小深度
        self.depth_max: float = 0.0  # 数据最大深度
        self.depth_min, self.depth_max = data_manager._get_depth_limits()           # 获取深度范围
        self.depth_position: float = self.depth_min                                 # 当前视图顶部深度位置
        self.window_size: float = 0.1*(self.depth_max-self.depth_min)               # 当前显示窗口的深度范围
        self.resolution: float = data_manager.get_logging_resolution()              # 深度采样分辨率（点间距） 这个是曲线的分辨率，主要是用来进行分类函数结果绘制的，其他的没啥用
        self.litho_width_config: Dict[int, float] = self.config_type['width_type']  # 岩性类型对应的显示宽度配置

        # 图形对象属性：存储 matplotlib 图形组件
        self.fig: Optional[plt.Figure] = None                                       # 主图形对象
        self.axs: Optional[List[plt.Axes]] = None                                   # 子图轴对象列表
        self.window_size_slider: Optional[Slider] = None                            # 窗口大小滑动条

        # 常规测井相关属性
        self.plots: List[Any] = []                                                  # 曲线绘图对象列表
        self.class_axes: List[plt.Axes] = []                                        # 分类数据子图轴列表

        # FMI 相关属性
        self.fmi_axes: List[plt.Axes] = []                                          # FMI图像子图轴列表
        self.fmi_images: List[Any] = []                                             # FMI图像对象列表，画图用的

        # NMR相关属性
        self.nmr_axes: List[plt.Axes] = []
        self.nmr_plots: List[Dict[str, Any]] = []  # 存储每个NMR道的绘图对象
        self.config_num_density: Dict[str, int] = {}   # 存放在不同显示窗长配置下，要显示几个NMR谱

        # 设置matplotlib字体
        self._setup_matplotlib_fonts()

    def _setup_matplotlib_fonts(self):
        """设置matplotlib字体 - 使用集中化的字体配置"""
        try:
            plt.rcParams['font.family'] = self.FONT_CONFIG['family']
            plt.rcParams['axes.unicode_minus'] = False
            self.logger.info("字体设置完成，使用配置: %s", self.FONT_CONFIG['family'])
        except Exception as e:
            self.logger.warning("字体设置失败: %s, 使用默认设置", e)

    def _calculate_subplot_count(self) -> int:
        """计算子图总数：FMI面板 + 曲线面板 + 分类面板 + NMR谱面板"""
        self.n_curve_panels, self.n_type_panels, self.n_fmi_panels, self.n_nmr_panels = self.data_manager.cal_plot_num()

        self.num_total = self.n_curve_panels+self.n_type_panels+self.n_fmi_panels+self.n_nmr_panels

        if self.num_total == 0:
            self.logger.warning("没有可显示的子图内容，请至少提供一种数据类型")

        return self.num_total

    def _setup_figure_layout(self, figure: Optional[plt.Figure], n_plots: int, has_legend: bool, figsize: Tuple[float, float]) -> None:
        """
        设置图形布局和子图排列

        布局逻辑：
        - 创建指定数量的子图，共享Y轴（深度轴）
        - 调整边距为图例和滑动条留空间
        - 子图间无间距（wspace=0）确保紧凑布局
        """
        if n_plots == 0:
            raise ValueError("没有可显示的子图内容")

        # 创建或重用图形对象
        if figure is None:
            # 创建新图形：1行n_plots列，共享Y轴，子图间无间距
            self.fig, self.axs = plt.subplots(1, n_plots, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.0})
        else:
            # 重用现有图形：清除内容后重新创建子图
            self.fig = figure
            self.fig.clear()
            self.axs = self.fig.subplots(1, n_plots, sharey=True, gridspec_kw={'wspace': 0.0})

        # 确保axs为列表（单子图时subplots返回单个Axes对象）
        if n_plots == 1:
            self.axs = [self.axs]

        # 根据是否有图例选择底部边距
        bottom_margin = (self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend else self.LAYOUT_CONFIG['no_legend_bottom_margin'])

        # 调整图形布局参数
        plt.subplots_adjust(
            left=self.LAYOUT_CONFIG['left_margin'],  # 左侧边距
            right=self.LAYOUT_CONFIG['right_margin'],  # 右侧边距（为滑动条留空间）
            bottom=bottom_margin,  # 底部边距
            top=1-self.LAYOUT_CONFIG['title_height'],  # 顶部边距
            wspace=0.0  # 子图间水平间距为0（紧密排列）
        )

    def _create_window_size_slider(self, has_legend: bool) -> None:
        """
        创建窗口大小滑动条

        滑动条功能：
        - 控制显示窗口的深度范围
        - 垂直方向，位于图形右侧
        - 范围从最小显示比例到完整深度范围
        """
        # 计算滑动条位置和尺寸
        bottom_margin = (
            self.LAYOUT_CONFIG['legend_bottom_margin'] if has_legend else self.LAYOUT_CONFIG['no_legend_bottom_margin'])
        slider_height = 1 - self.LAYOUT_CONFIG['title_height'] - bottom_margin

        # 创建滑动条轴对象（右侧垂直条）
        slider_ax = plt.axes([
            self.LAYOUT_CONFIG['right_margin'],  # x位置：右侧边距处
            bottom_margin,  # y位置：底部边距处
            self.LAYOUT_CONFIG['slider_width'],  # 宽度
            slider_height  # 高度：从底部到顶部
        ])

        # 计算滑动条数值范围
        depth_range = self.depth_max - self.depth_min
        min_window_size = depth_range * self.LAYOUT_CONFIG['min_window_size_ratio']  # 最小窗口大小
        max_window_size = depth_range  # 最大窗口大小（完整范围）
        initial_window_size = depth_range * 0.5  # 初始窗口大小（一半范围）
        self.window_size = initial_window_size  # 设置当前窗口大小

        # 创建滑动条对象
        self.window_size_slider = Slider(
            ax=slider_ax,
            label='',  # 空标签（使用文本标签代替）
            valmin=min_window_size,
            valmax=max_window_size,
            valinit=initial_window_size,
            orientation='vertical',     # 垂直方向
        )
        # 隐藏滑动条数值文本     # 不显示数值格式
        self.window_size_slider.valtext.set_visible(False)

        # 添加滑动条文本标签（旋转270度）
        slider_ax.text(0.5, 0.5, '窗口大小(m)', rotation=270, ha='center', va='center', transform=slider_ax.transAxes, fontsize=self.LAYOUT_CONFIG['tick_label_fontsize'])

    def _create_title_box(self, ax: plt.Axes, title: Any, color: str, index: int) -> None:
        """
        为子图创建标题框

        标题框设计：
        - 位于子图顶部中央
        - 浅灰色背景，细边框
        - 粗体文本，指定颜色
        - 高zorder确保显示在最上层
        """
        # 获取子图在图形中的位置（相对坐标）
        orig_pos = ax.get_position()

        # 使用配置中的标题参数
        title_bbox = [
            orig_pos.x0 + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.y0 + orig_pos.height + self.LAYOUT_CONFIG['title_margin'],
            orig_pos.width - 2 * self.LAYOUT_CONFIG['title_margin'],
            self.LAYOUT_CONFIG['title_height']
        ]

        # 创建标题背景矩形 - 使用配置中的透明度
        title_rect = Rectangle(
            (title_bbox[0], title_bbox[1]), title_bbox[2], title_bbox[3],
            transform=self.fig.transFigure,                             # 使用图形坐标变换
            facecolor=self.RENDERING_CONFIG['head_box_facecolor'],      # 浅灰色背景
            edgecolor=self.RENDERING_CONFIG['head_box_edgecolor'],      # 灰色边框
            linewidth=self.RENDERING_CONFIG['title_box_linewidth'],     # 使用配置的线宽
            clip_on=False,                                              # 不裁剪（允许显示在子图外）
            zorder=10,                                                  # 高层级确保显示在前面
        )
        self.fig.add_artist(title_rect)

        if isinstance(title, list):
            # 多部分标题
            n_parts = len(title)
            # part_width = title_bbox[2] / n_parts
            part_height = title_bbox[3] / n_parts

            for i, part_text in enumerate(title):
                # 计算每个部分的中心位置
                x_center = title_bbox[0] + title_bbox[2] / 2
                y_center = title_bbox[1] + (i+0.5)*part_height

                # 使用配置的颜色循环
                part_color = self.DEFAULT_MULTI_CURVE_COLORS[i % len(self.DEFAULT_MULTI_CURVE_COLORS)]
                # 创建单个标题部分文本
                text_obj = Text(
                    x_center, y_center, part_text,
                    fontsize=self.LAYOUT_CONFIG['title_fontsize'],  # 使用配置的字体大小
                    fontweight=self.FONT_CONFIG['weight_bold'],  # 使用配置的字重
                    color=part_color,
                    ha='center', va='center',
                    transform=self.fig.transFigure,
                    clip_on=False,
                    zorder=11
                )
                self.fig.add_artist(text_obj)
        else:
            # 单标题 # 单标题：使用传入的颜色
            title_text = Text(
                title_bbox[0] + title_bbox[2] / 2,
                title_bbox[1] + title_bbox[3] / 2,
                title,
                fontsize=self.LAYOUT_CONFIG['title_fontsize'],  # 使用配置的字体大小
                fontweight=self.FONT_CONFIG['weight_bold'],  # 使用配置的字重
                color=color,
                ha='center', va='center',
                transform=self.fig.transFigure,
                clip_on=False,
                zorder=11
            )
            self.fig.add_artist(title_text)

    def _plot_fmi_panel(self, ax: plt.Axes, image_data: np.ndarray, title: str, index: int) -> None:
        """
        绘制FMI图像面板

        支持图像格式：
        - 2D灰度图像：使用热力图色彩映射
        - 3D彩色图像：直接显示RGB或RGBA
        - 3D单通道：转换为2D显示
        """
        # 创建标题框
        self._create_title_box(ax, title, '#222222', index)
        fmi_depth = self.fmi_data_windows['depth']

        # 根据图像维度选择显示方法
        if len(image_data.shape) == 2:
            # 2D图像：使用热力图色彩映射
            img = ax.imshow(image_data, aspect='auto', cmap=self.config_fmi['color_map'], extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])

        elif len(image_data.shape) == 3 and image_data.shape[2] in [1, 3, 4]:
            # 3D图像：单通道转换为2D，多通道直接显示
            display_data = image_data if image_data.shape[2] != 1 else image_data[:, :, 0]
            img = ax.imshow(display_data, aspect='auto', extent=[0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])
        else:
            raise ValueError(f"不支持的图像维度: {image_data.shape}")

        # 保存图像对象用于后续更新
        self.fmi_images.append(img)

        # 设置坐标轴：隐藏X轴标签
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_xlabel('')

        # ax.set_ylim(fmi_depth[-1], fmi_depth[0])                # 设置Y轴范围
        ax.set_ylim(fmi_depth[-1], fmi_depth[0])

        # Y轴显示逻辑：只有第一个面板显示Y轴标签
        if index == 0:  # 第一个面板显示Y轴
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
        else:
            ax.tick_params(left=False, labelleft=False)  # 其他面板隐藏Y轴


    def _plot_all_fmi_panels(self) -> None:
        """绘制所有FMI图像面板"""
        self.fmi_axes = []  # 清空FMI轴列表
        self.fmi_images = []  # 清空图像对象列表

        if not self.fmi_data_windows:
            return  # 无FMI数据时直接返回

        # 为每个FMI图像创建面板
        for i, (image_data, title) in enumerate(zip(self.fmi_data_windows['image_data'], self.config_fmi['title_fmi'])):
            ax = self.axs[i]  # 使用前几个子图
            self.fmi_axes.append(ax)  # 记录FMI轴
            self._plot_fmi_panel(ax, image_data, title, i)

    def _plot_curve_panel(self, ax: plt.Axes, curve_item: Any, color: str, index: int, line_style: Any, line_width: Any) -> None:
        """
        绘制单个曲线面板

        曲线显示特性：
        - 深度在Y轴，曲线值在X轴
        - 自动计算合适的显示范围
        - 添加网格线提高可读性
        - 反转Y轴符合地质显示习惯
        """
        if isinstance(curve_item, list):
            # 修改：多曲线道 - 绘制多条曲线
            self._create_title_box(ax, curve_item, color, index)

            # 为每条曲线分配颜色
            for i, curve_col in enumerate(curve_item):
                curve_color = color[i]
                width = line_width[i] if isinstance(line_width, list) else line_width
                style = line_style[i] if isinstance(line_style, list) else line_style
                line, = ax.plot(self.logging_data_windows[curve_col].values, self.logging_data_windows[self.config_logging['depth_col']].values,
                                color=curve_color, linewidth=width, linestyle=style, label=curve_col)
                self.plots.append(line)

            # 计算多曲线的联合显示范围
            all_min, all_max = float('inf'), float('-inf')
            for curve_col in curve_item:
                # min_val, max_val = self._calculate_curve_display_limits(self.logging_data_windows[curve_col])
                min_val, max_val = self.data_manager._calculate_curve_display_limits(curve_col)
                all_min = min(all_min, min_val)
                all_max = max(all_max, max_val)

            ax.set_xlim(all_min, all_max)
        elif isinstance(curve_item, str):
            # 单曲线道 - 原有逻辑 创建标题框
            self._create_title_box(ax, curve_item, color, index)
            # 绘制曲线：X=曲线值，Y=深度
            line, = ax.plot(self.logging_data_windows[curve_item].values, self.logging_data_windows[self.config_logging['depth_col']].values,
                            color=color, linewidth=line_width, linestyle=line_style, label=curve_item)
            self.plots.append(line)  # 保存绘图对象
            # 设置X轴显示范围
            # min_val, max_val = self._calculate_curve_display_limits(self.logging_data_windows[curve_item])
            min_val, max_val = self.data_manager._calculate_curve_display_limits(curve_item)
            ax.set_xlim(min_val, max_val)
        else:
            raise ValueError(f"不支持的曲线项类型: {type(curve_item)}")

        # 设置坐标轴属性
        ax.grid(True, alpha=0.3)  # 添加半透明网格线

        # Y轴显示逻辑：无FMI时的第一个曲线面板显示Y轴
        if not self.fmi_axes and index == 0:  # 无FMI时的第一个曲线面板
            ax.set_ylabel('深度 (m)', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            # ax.set_ylim([self.depth_max, self.depth_min])  # 设置Y轴范围
        else:
            ax.tick_params(left=False, labelleft=False)  # 其他面板隐藏Y轴

    def _plot_all_curves(self) -> None:
        """绘制所有曲线面板"""
        self.plots = []  # 清空绘图对象列表

        if self.logging_data_windows is None or self.config_logging['curves_plot'] is None:
            self.logger.debug("无曲线数据，跳过曲线绘制")
            return

        # 计算曲线面板的起始索引（在FMI面板之后）
        start_index = self.n_fmi_panels

        # 为每条曲线创建面板
        for i, item in enumerate(self.config_logging['curves_plot']):
            ax_index = start_index + i
            color = self.config_logging['colors_plot'][i]
            line_style = self.config_logging['line_style_plot'][i]
            line_width = self.config_logging['line_width'][i]
            self._plot_curve_panel(self.axs[ax_index], item, color, ax_index, line_style, line_width)

    def _batch_render_classification(self, ax: plt.Axes, class_col: str, visible_data: pd.DataFrame) -> None:
        """
        批量渲染分类数据（性能优化版本）

        优化策略：
        - 使用PolyCollection批量绘制多边形，比逐个绘制矩形性能更高
        - 按分类值分组，相同颜色的矩形批量处理
        - 减少matplotlib绘图调用次数
        """
        if visible_data.empty:
            return  # 无可见数据时直接返回

        # 按分类值分组数据
        class_groups = visible_data.groupby(class_col)
        vertices_list = []  # 存储所有矩形的顶点
        colors_list = []  # 存储对应的颜色

        # 为每个分类值创建矩形
        for class_val, group in class_groups:
            if pd.isna(class_val) or class_val < 0:
                continue  # 跳过无效值

            class_int = int(class_val)
            # 获取该分类的显示宽度
            xmax = self.litho_width_config.get(class_int, 0.1)

            # 根据分类值选择颜色
            color = self.config_type['colors_type'][class_int]
            # color = self.DEFAULT_TYPE_COLORS[class_int % len(self.DEFAULT_TYPE_COLORS)]

            # 为每个深度点创建矩形
            for depth in group[self.config_logging['depth_col']]:
                # 计算矩形的上下边界（基于深度分辨率）
                y_bottom = depth - self.resolution / 2
                y_top = depth + self.resolution / 2

                # 定义矩形的四个顶点（左下→右下→右上→左上）
                vertices = [[0, y_bottom], [xmax, y_bottom], [xmax, y_top], [0, y_top]]
                vertices_list.append(vertices)
                colors_list.append(color)

        # 如果有矩形数据，批量绘制
        if vertices_list:
            poly_collection = PolyCollection(vertices_list,
                                             facecolors=colors_list,  # 填充颜色
                                             edgecolors='none',  # 无边框
                                             linewidths=0)  # 边框宽度0
            ax.add_collection(poly_collection)  # 添加到轴

    def _plot_all_classification_panels(self) -> None:
        """绘制所有分类数据面板"""
        self.class_axes = []  # 清空分类轴列表

        if not self.config_type['types_cols']:
            return  # 无分类数据时直接返回

        # 计算分类面板的起始索引（在FMI和曲线面板之后）
        base_index = self.n_fmi_panels + self.n_curve_panels + self.n_nmr_panels

        # 为每个分类列创建面板
        for i, col in enumerate(self.config_type['types_cols']):
            ax_idx = base_index + i  # 计算子图索引
            ax = self.axs[ax_idx]
            self.class_axes.append(ax)  # 记录分类轴
            self._plot_classification_panel(ax, col, ax_idx)

    def _plot_classification_panel(self, ax: plt.Axes, class_col: str, index: int) -> None:
        """
        绘制分类数据面板（初始绘制，非优化版本）
        用于初始显示，后续更新使用优化的批量渲染版本
        """
        self._create_title_box(ax, class_col, '#222222', index)

        # 为每个测井道绘制矩形
        for depth, class_val in zip(self.logging_data_windows[self.config_logging['depth_col']], self.logging_data_windows[class_col]):
            if pd.isna(class_val) or class_val < 0:
                continue  # 跳过无效值

            class_int = int(class_val)
            # 获取显示宽度
            xmax = self.litho_width_config.get(class_int, 0.1)

            # 绘制水平矩形条
            ax.axhspan(ymin=depth - self.resolution / 2,  # 下边界
                       ymax=depth + self.resolution / 2,  # 上边界
                       xmin=0, xmax=xmax,  # 左右边界
                       # facecolor=self.DEFAULT_TYPE_COLORS[class_int % len(self.DEFAULT_TYPE_COLORS)],
                       facecolor=self.config_type['colors_type'][class_int],
                       edgecolor='none')  # 无边框

        # 设置坐标轴属性
        ax.set_xticks([])  # 隐藏X轴刻度
        ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴

    def _plot_nmr_panel(self, ax: plt.Axes, nmr_index: int, panel_index: int) -> None:
        """绘制单个NMR谱面板 - 确保正确设置对数坐标轴"""
        title = self.config_nmr.get('nmr_title', [f'NMR_{i+1}' for i in range(10)])[nmr_index]

        # 创建标题框
        self._create_title_box(ax, title, '#000000', panel_index)

        # 设置坐标轴属性
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True, alpha=0.3)

        # 移除X轴的刻度线和刻度标签
        ax.tick_params(
            axis='x',  # 针对x轴
            which='both',  # 同时修改主刻度和次刻度
            bottom=False,  # 移除底部刻度线
            top=False,  # 移除顶部刻度线
            labelbottom=False  # 移除底部刻度标签
        )

        if self.config_nmr['x_logarithmic_scale']:
            # T2核磁谱设置
            ax.set_xscale('log')                            # T2时间使用对数坐标

        # 初始化对象池，避免频繁创建销毁
        nmr_plot_info = {
            'ax': ax,
            'nmr_index': nmr_index,
            'line_pool': [],  # 谱线对象池
            'fill_pool': [],  # 填充对象池
            'active_lines': [],  # 当前活动的谱线
            'active_fills': [],  # 当前活动的填充
            'max_pool_size': 50  # 对象池最大容量
        }
        self.nmr_plots.append(nmr_plot_info)

        # 预创建对象池
        self._initialize_nmr_object_pool(nmr_plot_info)

        # 隐藏Y轴标签（非第一个面板）
        if panel_index > 0:
            ax.tick_params(left=False, labelleft=False)

    def _initialize_nmr_object_pool(self, nmr_plot: Dict[str, Any]) -> None:
        """初始化NMR绘图对象池"""
        ax = nmr_plot['ax']
        max_pool_size = nmr_plot['max_pool_size']

        # 预创建谱线对象
        for i in range(max_pool_size):
            line, = ax.plot([], [], 'g-', linewidth=self.config_nmr['spectrum_config']['line_width'], alpha=self.config_nmr['spectrum_config']['fill_alpha'], visible=False)
            nmr_plot['line_pool'].append(line)

        # 预创建填充对象
        for i in range(max_pool_size):
            fill = ax.fill_between([], [], [], linewidth=self.config_nmr['spectrum_config']['line_width'], alpha=self.config_nmr['spectrum_config']['fill_alpha'], color=self.config_nmr['color_fill'], visible=False)
            nmr_plot['fill_pool'].append(fill)

    def _plot_all_nmr_panels(self) -> None:
        """绘制所有NMR谱面板"""
        self.nmr_axes = []
        self.nmr_plots = []
        self.sorted_depths_NMR = []

        if self.nmr_data_windows is None or not self.nmr_data_windows:
            return

        # 计算NMR面板的起始索引
        base_index = self.n_fmi_panels + self.n_curve_panels

        # 为每个NMR数据组创建面板
        for i in range(len(self.nmr_data_windows['nmr_data'])):
            ax_idx = base_index + i
            if ax_idx < len(self.axs):
                ax = self.axs[ax_idx]
                self.nmr_axes.append(ax)
                self._plot_nmr_panel(ax, i, ax_idx)

    def _update_nmr_display(self, ) -> None:
        if not self.nmr_plots or not self.nmr_data_windows:
            return

        if 'nmr_data' in self.nmr_data_windows and self.nmr_data_windows['nmr_data']:
            nmr_data_list = self.nmr_data_windows['nmr_data']
            nmr_depth = self.nmr_data_windows['depth']
        else:
            # 该NMR道在当前深度范围内无数据
            return

        for i, nmr_plot in enumerate(self.nmr_plots):
            ax = nmr_plot['ax']

            # 隐藏所有当前活动的对象
            self._hide_all_active_objects(nmr_plot)

            nmr_data = nmr_data_list[i]

            # 计算需要的对象数量
            # needed_objects = self.nmr_data_windows['nmr_data'].shape[0]
            needed_objects = nmr_data.shape[0]

            # 确保对象池足够大
            if needed_objects > nmr_plot['max_pool_size']:
                self._expand_object_pool(nmr_plot, needed_objects)

            scale_factor = self.config_nmr['plot_amplitude_scaling']*(nmr_depth.max() - nmr_depth.min())/20
            # # nmr数据的校正归一化
            nmr_data = nmr_data * scale_factor

            # 为每个深度的NMR数据绘制谱图
            # for i, (depth, nmr_data) in enumerate(nmr_data_group.items()):
            for i in range(nmr_data.shape[0]):
                if i >= len(nmr_plot['line_pool']):
                    break  # 对象池不足，跳过

                NMR_X = np.linspace(0.001, nmr_data.shape[1], nmr_data.shape[1])
                NMR_Y = nmr_data[i, :]

                # 归一化振幅并添加深度偏移
                normalized_NMR_Y = NMR_Y
                y_values = nmr_depth[i] - normalized_NMR_Y

                # 创建基线（深度水平线）
                baseline = np.full_like(y_values, nmr_depth[i])

                # 复用对象池中的对象
                line = nmr_plot['line_pool'][i]
                fill = nmr_plot['fill_pool'][i]

                # 更新谱线数据
                line.set_data(NMR_X, y_values)
                line.set_visible(True)
                nmr_plot['active_lines'].append(line)

                # 修复填充对象问题：先检查是否存在再移除
                if fill in ax.collections:
                    fill.remove()
                # 只有存在时才移除
                elif hasattr(fill, 'collections') and fill.collections:
                    for coll in fill.collections:
                        if coll in ax.collections:
                            coll.remove()

                # 创建新的填充
                new_fill = ax.fill_between(NMR_X, baseline, y_values, alpha=self.config_nmr['spectrum_config']['fill_alpha'], color=self.config_nmr['color_fill'], linewidth=0)
                nmr_plot['fill_pool'][i] = new_fill
                nmr_plot['active_fills'].append(new_fill)

    def _hide_all_active_objects(self, nmr_plot: Dict[str, Any]) -> None:
        """隐藏所有当前活动的绘图对象 - 修复移除错误"""
        ax = nmr_plot['ax']

        # 隐藏谱线
        for line in nmr_plot['active_lines']:
            line.set_visible(False)
        nmr_plot['active_lines'] = []

        # 修复填充移除：安全地移除存在的对象
        for fill in nmr_plot['active_fills']:
            try:
                # 检查填充对象是否还在轴上
                if fill in ax.collections:
                    fill.remove()
                elif hasattr(fill, 'collections') and fill.collections:
                    # 处理fill_between返回的PolyCollection
                    for coll in fill.collections:
                        if coll in ax.collections:
                            coll.remove()
            except (ValueError, AttributeError) as e:
                # 如果对象已经不存在，忽略错误
                self.logger.debug(f"移除填充对象时出错: {e}")
                continue
        nmr_plot['active_fills'] = []

    def _expand_object_pool(self, nmr_plot: Dict[str, Any], new_size: int) -> None:
        """扩展对象池大小"""
        ax = nmr_plot['ax']
        current_size = len(nmr_plot['line_pool'])

        # 扩展谱线对象池
        for i in range(current_size, new_size):
            # line, = ax.plot([], [], 'g-', linewidth=0.8, alpha=0.9, visible=False)
            line, = ax.plot([], [], 'g-', linewidth=self.config_nmr['spectrum_config']['line_width'], alpha=self.config_nmr['spectrum_config']['fill_alpha'], visible=False)
            nmr_plot['line_pool'].append(line)

        # 扩展填充对象池
        for i in range(current_size, new_size):
            # fill = ax.fill_between([], [], [], alpha=0.5, color='green', linewidth=0, visible=False)
            fill = ax.fill_between([], [], [], linewidth=self.config_nmr['spectrum_config']['line_width'], alpha=self.config_nmr['spectrum_config']['fill_alpha'], color=self.config_nmr['color_fill'], visible=False)
            nmr_plot['fill_pool'].append(fill)

        nmr_plot['max_pool_size'] = new_size

    def _optimize_fmi_rendering(self) -> None:
        """
        FMI图像渲染优化：预处理图像数据提高显示性能

        优化内容：
        - 将浮点图像数据归一化到0-255并转换为uint8
        - 减少图像数据传输和内存占用
        - 提高imshow函数的渲染效率
        """
        if not self.fmi_data_windows:
            return  # 无FMI数据时直接返回

        optimized_images = []

        # 处理每个FMI图像
        for i, image_data in enumerate(self.fmi_data_windows['image_data']):
            if image_data is None:
                optimized_images.append(None)
                continue

            # 1. 数据类型转换优化
            if image_data.dtype != np.uint8:
                # 归一化到0-255
                # if image_data.size > 0 and np.max(image_data) > np.min(image_data):
                #     image_norm = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
                #     image_opt = image_norm.astype(np.uint8)
                # else:
                #     image_opt = np.full_like(image_data, 128, dtype=np.uint8)
                image_opt = np.clip(image_data, 0, 256).astype(np.uint8)
            else:
                image_opt = image_data

            optimized_images.append(image_opt)

        # 更新优化后的图像数据
        self.fmi_data_windows['image_data'] = optimized_images

    def _create_legend_panel(self, legend_dict: Dict[int, str]) -> None:
        """
        创建图例面板

        图例设计：
        - 位于图形底部中央
        - 多列布局，自动调整列数
        - 半透明背景，细边框
        - 使用岩性对应的颜色显示
        """
        if not legend_dict:
            return  # 无图例数据时直接返回

        n_items = len(legend_dict)
        # 计算图例尺寸：宽度基于项目数，固定高度
        legend_height, legend_width = 0.02, min(0.8, n_items * 0.15)

        # 创建图例轴对象（底部中央，无边框）
        legend_ax = plt.axes([0.5 - legend_width / 2, 0.01, legend_width, legend_height], frameon=False)
        legend_ax.set_axis_off()  # 隐藏坐标轴

        # 准备图例句柄和标签
        handles, labels = [], []
        for key in sorted(legend_dict.keys()):
            # 创建颜色块作为图例句柄
            patch = Rectangle((0, 0), 1, 1,
                              facecolor=self.config_type['colors_type'][int(key)],
                              edgecolor='black',
                              linewidth=self.config_type['display_config']['border_width']
                              )
            handles.append(patch)
            labels.append(legend_dict[key])

        # 创建图例：中央位置，自动列数，半透明背景
        # legend = legend_ax.legend(handles, labels, loc='center', ncol=min(n_items, 6), frameon=True, framealpha=self.LAYOUT_CONFIG['legend_box_alpha'], fontsize=self.LAYOUT_CONFIG['legend_fontsize'])
        legend = legend_ax.legend(handles, labels, loc='center', ncol=self.config_type['display_config']['legend_columns'], frameon=True,
                                  framealpha=self.config_type['display_config']['pattern_alpha'],
                                  fontsize=self.config_type['display_config']['legend_fontsize']
                                  )

        # 设置图例框样式
        frame = legend.get_frame()
        # frame.set_facecolor(self.LAYOUT_CONFIG['legend_facecolor'])  # 浅灰色背景
        frame.set_facecolor(self.config_type['display_config']['legend_facecolor'])  # 浅灰色背景

    def _on_window_size_change(self, val: float) -> None:
        """
        窗口大小变化事件处理

        处理逻辑：
        1. 更新窗口大小
        2. 调整深度位置确保不超出有效范围
        3. 触发显示更新
        """
        if val <= 0:
            return  # 避免无效值

        self.window_size = val  # 更新窗口大小

        # 计算最大有效深度位置（确保窗口不超出数据范围）
        max_valid_position = self.depth_max - self.window_size
        if self.depth_position > max_valid_position:
            self.depth_position = max_valid_position  # 调整位置

        self._update_display()  # 更新显示

    def _on_mouse_scroll(self, event) -> None:
        """
        鼠标滚轮事件处理：实现深度滚动

        滚动逻辑：
        - 向上滚动：向浅部（小深度）滚动
        - 向下滚动：向深部（大深度）滚动
        - 步长为窗口大小的10%
        """
        # 检查事件是否发生在子图内
        if event.inaxes not in self.axs:
            return

        # 检查是否在正确的坐标系中
        if not any(event.inaxes == ax for ax in self.axs):
            return

        # 计算滚动步长（窗口大小的10%）
        step = self.window_size * self.LAYOUT_CONFIG['scroll_step_ratio']

        # 根据滚轮方向计算新位置
        if event.button == 'up':
            new_position = self.depth_position - step  # 向上滚动：向浅部
        elif event.button == 'down':
            new_position = self.depth_position + step  # 向下滚动：向深部
        else:
            return  # 非滚轮事件忽略

        # 限制位置范围
        new_position = max(self.depth_min, min(new_position, self.depth_max - self.window_size))

        if new_position != self.depth_position:
            self.depth_position = new_position
            self._update_display()


    def _update_display(self) -> None:
        """
        更新显示内容（核心刷新函数）

        刷新流程：
        1. 计算当前显示范围
        2. 更新所有子图的Y轴范围
        3. 更新分类面板内容
        4. 更新FMI图像显示
        5. 更新深度信息显示
        6. 重绘图形
        7. 记录性能数据
        """
        start_time = time.time()  # 记录开始时间


        # 计算当前显示的深度范围
        top_depth = self.depth_position
        bottom_depth = self.depth_position + self.window_size

        self.logging_data_windows = self.data_manager.get_visible_logging_data(top_depth, bottom_depth)
        self.fmi_data_windows = self.data_manager.get_visible_fmi_data(top_depth, bottom_depth)

        # print(self.fmi_data_windows['depth'].max(), self.fmi_data_windows['depth'].min(), self.fmi_data_windows['image_data'][0].shape)

        nmr_plot_density = self.config_nmr['plot_density']/(self.window_size)
        self.nmr_data_windows = self.data_manager.get_visible_nmr_data(self.depth_position, self.depth_position+self.window_size, nmr_plot_density)

        # 更新所有子图的Y轴范围
        for ax in self.axs:
            ax.set_ylim(top_depth, bottom_depth)            # 注意：matplotlib中Y轴从上到下

        # 条件更新各显示组件
        if self.logging_data_windows is not None:
            self._update_curve_panels()  # 更新曲线数据
        self.logger.debug("渲染常规测井完成: %.1fms", (time.time() - start_time) * 1000)
        start_time = time.time()

        if self.logging_data_windows is not None and self.config_type['types_cols']:
            self._update_classification_panels()
        self.logger.debug("渲染分类结果完成: %.1fms", (time.time() - start_time) * 1000)
        start_time = time.time()

        if self.fmi_data_windows is not None:
            self._update_fmi_display()
        self.logger.debug("渲染电成像完成: %.1fms", (time.time() - start_time) * 1000)
        start_time = time.time()

        if self.nmr_data_windows is not None:
            self._update_nmr_display()
        self.logger.debug("渲染核磁完成: %.1fms", (time.time() - start_time) * 1000)
        start_time = time.time()

        self._update_depth_indicator(top_depth, bottom_depth)
        self.logger.debug("渲染深度指示器完成: %.1fms", (time.time() - start_time) * 1000)
        start_time = time.time()

        # 请求图形重绘
        self.fig.canvas.draw_idle()

        render_time = (time.time() - start_time) * 1000
        self.logger.debug("渲染完成: %.1fms", render_time)

        # ✅ 新增：确保y轴方向正确
        self._ensure_correct_yaxis_direction()

        # 请求图形重绘
        self.fig.canvas.draw_idle()


    # 强制反转Y轴
    def _ensure_correct_yaxis_direction(self):
        """确保所有子图的y轴方向正确（深度从上到下递增）"""
        for ax in self.axs:
            # 获取当前y轴范围
            ymin, ymax = ax.get_ylim()

            # 如果当前范围是反的，就纠正
            if ymin < ymax:
                ax.set_ylim(ymax, ymin)  # 交换上下限

            # 确保y轴反转（从上到下递增）
            if not ax.yaxis_inverted():
                ax.invert_yaxis()

    def _update_curve_panels(self) -> None:
        """
        修复：更新常规测井曲线面板的数据
        确保曲线数据随深度范围变化而更新
        """
        if self.logging_data_windows is None or not self.plots:
            self.logger.debug("无曲线数据或绘图对象，跳过曲线更新")
            return

        # 获取当前深度数据
        depth_col = self.config_logging['depth_col']
        if depth_col not in self.logging_data_windows.columns:
            self.logger.warning(f"深度列 {depth_col} 不存在于数据中")
            return

        depth_data = self.logging_data_windows[depth_col].values

        # 遍历所有曲线绘图对象并更新数据
        plot_index = 0
        curve_configs = self.config_logging['curves_plot']

        for i, curve_item in enumerate(curve_configs):
            if plot_index >= len(self.plots):
                break  # 防止索引越界

            if isinstance(curve_item, list):
                # 多曲线道：更新该道内的所有曲线
                for curve_col in curve_item:
                    if plot_index < len(self.plots) and curve_col in self.logging_data_windows.columns:
                        line = self.plots[plot_index]
                        curve_values = self.logging_data_windows[curve_col].values

                        # ✅ 关键修复：更新线对象的数据
                        line.set_data(curve_values, depth_data)
                        plot_index += 1
                        # 记录调试信息
                        self.logger.debug(f"更新曲线 {curve_col}: {len(curve_values)} 个数据点")
            elif isinstance(curve_item, str):
                # 单曲线道：更新单条曲线
                if curve_item in self.logging_data_windows.columns:
                    line = self.plots[plot_index]
                    curve_values = self.logging_data_windows[curve_item].values

                    # ✅ 关键修复：更新线对象的数据
                    line.set_data(curve_values, depth_data)
                    plot_index += 1

                    self.logger.debug(f"更新曲线 {curve_item}: {len(curve_values)} 个数据点")

        self.logger.debug(f"共更新 {plot_index} 条曲线")


    def _update_classification_panels(self,) -> None:
        """
        更新分类面板显示

        更新策略：
        1. 清除现有内容
        2. 获取可见范围内的数据
        3. 使用批量渲染方法重新绘制
        """
        if not self.class_axes:
            return  # 无分类面板时直接返回

        # 更新每个分类面板
        for i, (ax, col) in enumerate(zip(self.class_axes, self.config_type['types_cols'])):
            ax.clear()  # 清除现有内容

            # 重新设置坐标轴属性
            ax.set_xticks([])  # 隐藏X轴
            ax.tick_params(left=False, labelleft=False)  # 隐藏Y轴
            ax.set_ylim(self.depth_position, self.depth_position + self.window_size)  # 设置Y轴范围

            # 使用批量渲染方法绘制分类数据
            self._batch_render_classification(ax, col, self.logging_data_windows)


    def _update_fmi_display(self, ) -> None:
        """更新FMI图像显示（使用缓存优化）"""
        if not self.fmi_data_windows or not self.fmi_images:
            return

        # 获取当前窗口的深度范围
        fmi_depth = self.fmi_data_windows['depth']

        # 更新每个FMI图像
        for i, (img, image_data) in enumerate(zip(self.fmi_images, self.fmi_data_windows['image_data'])):
            # 更新图像数据
            if len(image_data.shape) == 2:
                img.set_data(image_data)
                # 关键修复：更新extent参数，使用当前窗口的深度范围
                img.set_extent([0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])
            elif len(image_data.shape) == 3:
                display_data = image_data if image_data.shape[2] != 1 else image_data[:, :, 0]
                img.set_data(display_data)
                # 关键修复：更新extent参数
                img.set_extent([0, image_data.shape[1], fmi_depth[-1], fmi_depth[0]])

            if self.config_fmi['auto_scale']:
                # 确保图像自动缩放,这个会自适应将图像从图像的"最小值-最大值"归一化到"0-256"
                img.set_clim(vmin=np.min(image_data), vmax=np.max(image_data))

    def _update_depth_indicator(self, top_depth: float, bottom_depth: float) -> None:
        """
        更新深度指示器：显示当前视图的深度范围和窗口大小

        位置：图形右下角，半透明背景确保可读性
        """
        # 生成指示器文本
        indicator_text = (f" 深度[{self.depth_min:.2f}({top_depth:.2f}-{bottom_depth:.2f}){self.depth_max:.2f}] | "
                          f"窗口: {self.window_size:.2f} m ")

        if hasattr(self, '_depth_indicator'):
            # 更新现有文本对象
            self._depth_indicator.set_text(indicator_text)
        else:
            # 创建新文本对象（首次调用时）
            self._depth_indicator = self.fig.text(
                0.99, 0.01, indicator_text,  # 位置：右下角
                ha='right', va='bottom', fontsize=self.LAYOUT_CONFIG['depth_indicator_fontsize'],  # 对齐和字体
                bbox=dict(facecolor=self.LAYOUT_CONFIG['depth_indicator_facecolor'], alpha=self.LAYOUT_CONFIG['depth_indicator_alpha'], boxstyle='round,pad=0.1')  # 半透明背景框
            )

    def visualize(self,
                  figsize: Tuple[float, float] = (16, 10),
                  figure: Optional[plt.Figure] = None) -> None:
        """
        主可视化函数：完整的测井数据可视化流程

        执行流程：
        1. 参数预处理和验证
        2. 数据初始化和预处理
        3. 图形设置和布局
        4. 数据绘制和优化
        5. 交互功能设置
        6. 初始显示和性能监控
        """
        try:
            self.logger.info("开始测井数据可视化")
            start_time = time.time()  # 记录开始时间

            # ========== 获取待显示段数据 ==========
            self.logging_data_windows = self.data_manager.get_visible_logging_data(self.depth_position, self.depth_position+self.window_size)
            self.fmi_data_windows = self.data_manager.get_visible_fmi_data(self.depth_position, self.depth_position+self.window_size, self.config_fmi['pix_density_y'])
            nmr_plot_density = self.config_nmr['plot_density']/(self.window_size)
            self.nmr_data_windows = self.data_manager.get_visible_nmr_data(self.depth_position, self.depth_position+self.window_size, nmr_plot_density)

            # ========== 图形设置 ==========
            n_plots = self._calculate_subplot_count()                               # 计算子图总数
            has_legend = bool(self.config_type['show_legend'])                      # 检查是否有图例
            self._setup_figure_layout(figure, n_plots, has_legend, figsize)         # 设置布局
            self._create_window_size_slider(has_legend)  # 创建滑动条

            # ========== 优化和绘制 ==========
            self._optimize_fmi_rendering()  # FMI图像优化
            self._plot_all_fmi_panels()  # 绘制FMI面板
            self._plot_all_curves()  # 绘制曲线面板
            self._plot_all_classification_panels()  # 绘制分类面板
            self._plot_all_nmr_panels()         # 绘制NMR核磁谱面板

            # ========== 交互功能 ==========
            self.window_size_slider.on_changed(self._on_window_size_change)  # 滑动条回调
            self.fig.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)  # 滚轮事件
            self._create_legend_panel(self.config_type['legend_dict'])  # 创建图例

            # ========== 初始显示 ==========
            self._update_display()  # 执行首次显示更新

            # ========== 性能统计 ==========
            total_time = time.time() - start_time
            self.logger.info("可视化完成，耗时: %.2fs", total_time)
            plt.show()  # 显示图形

        except Exception as e:
            # ========== 异常处理 ==========
            self.logger.error("可视化失败: %s", str(e))
            if self.fig:
                plt.close(self.fig)  # 关闭图形释放资源
            raise  # 重新抛出异常

    def get_plot_config(self):
        return self.config_logging, self.config_fmi, self.config_nmr, self.config_type


if __name__ == '__main__':
    work_well = DATA_WELL(path_folder=r'F:\logging_workspace\FY1-15')

    path_list_logging = work_well.get_path_list_logging()
    path_list_fmi = work_well.get_path_list_fmi()
    path_list_table = work_well.get_path_list_table()

    print(path_list_logging)
    print(path_list_fmi)
    print(path_list_table)

    logging_data = work_well.combine_logging_table(logging_key=path_list_logging[0], table_key=path_list_table[0],
                                                   curve_names_logging=['DEPTH', 'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA']
                                                   )

    print(logging_data.describe())
    print(logging_data.head(10))
    COLS_ALL = logging_data.columns.to_list()
    print(f"总列数: {len(COLS_ALL)}")
    print(f"所有列名: {COLS_ALL}")
    print(work_well.get_table_replace_dict())

    data_fmi, depth_fmi = work_well.get_FMI(key=path_list_fmi[0])
    # data_nmr, depth_nmr = work_well.get_FMI(key=path_list_fmi[-1])
    data_nmr, depth_nmr = work_well.get_FMI(key=path_list_fmi[0])
    # print(data_fmi.shape, depth_fmi.shape, data_nmr.shape, depth_nmr.shape)
    # print(depth_fmi[0], depth_fmi[-1])
    # print(depth_nmr[0], depth_nmr[-1])


    LDM = LoggingDataManager(
        logging_data=logging_data,
        # logging_data=pd.DataFrame(),
        fmi_data={'depth': depth_fmi, 'image_data': [data_fmi]},
        nmr_data={'depth': depth_nmr, 'nmr_data': [data_nmr]}
    )
    # print(LDM._get_depth_limits())

    initial_stats = LDM.get_performance_stats()
    num_iterations = 3

    # config_plot = LDM.plot_config_check()
    # for i, key in enumerate(config_plot):
    #     print(key)
    #     print(config_plot[key])

    config_logging ={
        'depth_col': 'DEPTH',  # depth深度列的列名配置
        'curves_plot': ['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', ['COR_MEAN_DYNA', 'ASM_MEAN_DYNA']],  # 哪些曲线需要进行绘制,在一个list中代表要绘制到一个道中
        'colors_plot': ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', ['#00FFFF', '#FFA500']],  # 颜色配置
        'line_style_plot': ['-', '-', '-', '-', ['-', '-.']],  # 线宽配置
        'line_width': [1.0, 1.0, 1.0, 1.0, [1.5, 0.8]],  # 线宽配置
        'axis_config': []
     }

    config_fmi: Dict[str, Any] = {
        'color_map': 'rainbow',  # 电成像颜色绘制配置
        'scale_range': [0, 256],  # 电成像像素缩放配置
        'title_fmi': ['FMI_1']  # 电成像的绘图Title配置
    }

    config_nmr: Dict[str, Any] = {
        'plot_density': 40,  # 每窗口绘制的密度
        'plot_amplitude_scaling': 0.2,  # 折线幅度的缩放因子
        'x_logarithmic_scale': True,  # x方向是否进行对数刻度
        'color_fill': 'green',  # 折线填充颜色配置
        'nmr_title': ['NMR_1'],
        'spectrum_config': {'line_style': '-', 'line_width': 0.8, 'fill_alpha': 0.5, 'baseline_visible': True},     # 谱绘制的配置
        'axis_config': {'x_axis_label': 'T2 Time (ms)', 'y_axis_label': 'Amplitude', 'show_grid': True, 'log_ticks': [0.1, 1, 10, 100, 1000]}       # x轴配置
    }

    config_type: Dict[str, Any] = {
        'types_cols': [],  # 都是那些列需要进行分类的绘制
        'show_legend': True,
        'colors_type': {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#96CEB4', 4: '#FECA57'},  # 相对应的颜色设置
        'width_type': {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0},  # 相对应的宽度设置
        'legend_dict': {0: 'Sandstone', 1: 'Mudstone', 2: 'Shale', 3: 'Limestone', 4: 'Dolomite'},  # 分类的Legend配置
        'display_config': {'legend_columns': 4, 'border_visible': False, 'border_width': 0.5},      #
        'pattern_config': {'use_patterns': True, 'pattern_alpha': 0.3, 'hatch_patterns': ['/', '\\', '|', '-', '+', 'x', 'o', 'O']}
    }

    LDM.plot_config_check(config_logging=config_logging, config_fmi=config_fmi, config_nmr=config_nmr, config_type=config_type)

    well_viewer = WellLogVisualizer(LDM, config_logging=config_logging, config_fmi=config_fmi, config_nmr=config_nmr, config_type=config_type)

    # print(LDM.get_logging_resolution())
    # print(LDM.cal_plot_num())

    well_viewer.visualize()