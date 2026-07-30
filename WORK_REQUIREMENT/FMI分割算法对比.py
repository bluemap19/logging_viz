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

# import warnings
# warnings.filterwarnings("ignore", message=".*TripleDES.*")
# warnings.filterwarnings("ignore", message=".*Blowfish.*")

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

    # 图像二值化方法选择
    target_fmi = copy.deepcopy(fmi_dyna)
    target_depth = copy.deepcopy(depth_dyna)

    # 图像二值化，先开后闭使目标区域数量更多
    result = cal_fmis_segmentation(imgs=[target_fmi.copy()], depth=target_depth, windows=400, step=100,
                                   # post_method=['open', 'close', ],
                                   post_method=['open'],
                                   method_configs=[
                                    ('tophat_otsu', None),
                                    ('otsu', None),
                                    ('adaptive', None),
                                    ('kmeans', None),
                                    ('gmm', None),
                                    ('wavelet', None),
                                ])

    fmi_list = [255-fmi_dyna, 255-fmi_stat]
    title_fmi_list = ['dyna', 'stat']
    for method in list(result.keys()):
        print('current binary method is :', method)
        print('result image shape is :', result[method][0].shape, ',result data type is :', type(result[method]))

        mask_origin = copy.deepcopy(result[list(result.keys())[list(result.keys()).index(method)]][0])
        fmi_list.append(255-mask_origin)
        title_fmi_list.append('mask_{}'.format(method))
        # 黄铁矿区域选择，主要需要设置黄铁矿显示的面积阈值
        # total_pixels, target_mask = count_small_target_areas(result[list(result.keys())[0]][0], area_threshold=80)
        total_pixels, target_mask = count_target_areas(mask_origin, target_config={
            'area_range': (1, 160),
            'width_range': (0, 30),
            'height_range': (0, 30),
            'perimeter_range': (0.0, 10 ** 6),
            'area_perimeter_ratio_range': (0.0, 8),       # 一般是 height_range/4 或者是 width_range/4 ，范围越小越接近长条形状，裂缝形状，越大越接近圆形
        })
        fmi_list.append(255-target_mask)
        title_fmi_list.append('target_{}'.format(method))

    LDM = LoggingDataManager(
        fmi_data={'depth': target_depth, 'image_data': fmi_list},
    )

    well_viewer = WellLogVisualizer(LDM,
        config_fmi = {'color_map': 'hot', 'title_fmi':title_fmi_list}
    )

    config_logging, config_fmi, config_nmr, config_type, config_core = well_viewer.get_plot_config()
    print(config_logging, '\n', config_fmi, '\n', config_nmr, '\n', config_type)

    well_viewer.visualize(figsize=(24, 12))

