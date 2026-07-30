import copy

from src_fmi.fmi_data_save import fmi_data_save
from src_fmi.fmi_pyrite_content import cal_fmis_pyrite_content, count_small_target_areas, count_target_areas
from src_fmi.fmi_segmentation import cal_fmis_segmentation
from src_fmi.image_operation import show_Pic
from src_logging.logging_combine import combine_logging_data
from src_logging.logging_interpolation import ConventionalLogInterpolator
from src_plot.well_logging_viz.data_manager import LoggingDataManager
from src_plot.well_logging_viz.data_visulization import WellLogVisualizer
from src_well_data.data_logging_well import DATA_WELL


if __name__ == '__main__':
    # 井设置
    # path_well = r'Z:\logging_workspace\塬22'
    path_well = r'Z:\logging_workspace\姬119H2'
    data_well = DATA_WELL(path_well)

    print('well_path is :'+data_well.well_path)
    print(data_well.well_summary())
    for _, key in enumerate(data_well.well_summary()):
        print(key, data_well.well_summary()[key])

    # 电成像数据读取
    # fmi_dyna, depth_dyna = data_well.get_FMI(key='Z:\\logging_workspace\\塬22\\塬22_FMI_DYNA.txt', depth=[2670.002, 2680])
    # fmi_stat, depth_stat = data_well.get_FMI(key='Z:\\logging_workspace\\塬22\\塬22_FMI_STAT.txt', depth=[2670.002, 2680])
    fmi_dyna, depth_dyna = data_well.get_FMI(key='Z:\\logging_workspace\\姬119H2\\姬119H2_FMI_DYNA.txt')
    fmi_stat, depth_stat = data_well.get_FMI(key='Z:\\logging_workspace\\姬119H2\\姬119H2_FMI_STAT.txt')
    print('data fmi_dyna shape is:', fmi_dyna.shape, 'data depth shape is :', depth_dyna.shape, 'type fmi_dyna is :', type(fmi_stat))
    print('data fmi_stat shape is:', fmi_stat.shape, 'data depth shape is :', depth_stat.shape, 'type fmi_stat is :', type(fmi_stat))
    # 常规测井数据读取
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\塬22\\塬22_normal_logging_2.csv')
    data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\姬119H2_logging_data.csv')
    # 常规测井无效数据归
    data_logging[data_logging<-1000] = 0
    print(data_logging.describe())

    # 图像二值化方法选择
    target_fmi = copy.deepcopy(fmi_dyna)
    target_depth = copy.deepcopy(depth_dyna)

    # 图像二值化，先开后闭使目标区域数量更多
    result = cal_fmis_segmentation(imgs=[target_fmi.copy()], depth=target_depth, windows=400, step=100, post_method=['open', 'close', ], method_configs=[
        # 方法、图像预处理方式：bilateral、median、gaussian
        ('tophat_otsu', None),
        ('otsu', None),
        ('adaptive', None),
        ('kmeans', None),
        ('gmm', None),
        ('wavelet', None),
    ])

    fmi_list = [255-fmi_dyna, 255-fmi_stat]
    title_fmi_list = ['dyna', 'stat']
    curves_name_pyrite_result_list = []
    area_limit = 120
    width_limit = 25
    height_limit = 25
    perimeter_range = (0.0, 10 ** 6)
    area_perimeter_ratio_range = (0.0, 8)
    for method in list(result.keys()):
        print('current binary method is :', method)
        print('result image shape is :', result[method][0].shape, ',result data type is :', type(result[method]))

        mask_origin = copy.deepcopy(result[list(result.keys())[list(result.keys()).index(method)]][0])
        fmi_list.append(255-mask_origin)
        title_fmi_list.append('M_{}'.format(method))
        # 黄铁矿区域选择，主要需要设置黄铁矿显示的面积阈值
        # total_pixels, target_mask = count_small_target_areas(result[list(result.keys())[0]][0], area_threshold=80)
        total_pixels, target_mask = count_target_areas(mask_origin, target_config={
            'area_range': (1, area_limit),
            'width_range': (0, width_limit),
            'height_range': (0, height_limit),
            'perimeter_range': perimeter_range,
            'area_perimeter_ratio_range': area_perimeter_ratio_range,       # 一般是 height_range/4 或者是 width_range/4 ，范围越小越接近长条形状，裂缝形状，越大越接近圆形
        })
        fmi_list.append(255-target_mask)
        title_fmi_list.append('T_{}'.format(method))

        # 黄铁矿含量计算
        result_pyrite = cal_fmis_pyrite_content(
            depth_data=target_depth,
            list_fmis=[target_mask],
            config_windows={'window_length': 300, 'step': 5, 'depth_col_name': 'DEPTH', 'curves_names_list': [method + '_fmi']},
        )

        print('result_pyrite shape is :', result_pyrite.shape, ', type is', type(result_pyrite))
        print(result_pyrite.describe())

        # # data_logging的插值处理
        # interpolator = ConventionalLogInterpolator(method='pchip')
        # data_logging = interpolator.interpolate_logs(data_logging, depth_col='#DEPTH', target_length=result_pyrite.shape[0])

        # 合并数据，计算的黄铁矿含量 + 其他矿物含量结果
        data_logging = combine_logging_data(
            data_main=data_logging,
            data_vice=[result_pyrite],
            drop=True
        )
        print(data_logging.describe())

        curves_name_pyrite_result_list.append(method + '_fmi')

    data_logging.to_excel(path_well+'\\'+f'logging+电成像计算黄铁矿_area{area_limit}_width{width_limit}_height{height_limit}.xlsx', index=False, sheet_name='0')

    LDM = LoggingDataManager(
        logging_data=data_logging,
        fmi_data={'depth': target_depth, 'image_data': fmi_list},
    )

    well_viewer = WellLogVisualizer(
        LDM,
        config_logging={'curves_plot' : ['GR', 'DWFE_INCP', 'DWSI_INCP', 'PYRITE_QE']+curves_name_pyrite_result_list},
        config_fmi = {'color_map': 'hot', 'title_fmi':title_fmi_list}
    )

    config_logging, config_fmi, config_nmr, config_type = well_viewer.get_plot_config()
    print(config_logging, '\n', config_fmi, '\n', config_nmr, '\n', config_type)

    well_viewer.visualize(figsize=(24, 12))

