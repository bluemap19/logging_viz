from src_fmi.fmi_data_save import fmi_data_save
from src_fmi.fmi_pyrite_content import cal_fmis_pyrite_content, count_small_target_areas
from src_fmi.fmi_segmentation import cal_fmis_segmentation
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
    path_well = r'F:\logging_workspace\塬22'
    data_well = DATA_WELL(path_well)

    print(data_well.well_path)
    print(data_well.well_summary())
    print(data_well.well_summary()['paths_fmi'])

    # 电成像数据读取
    fmi_dyna, depth_dyna = data_well.get_FMI(key=data_well.well_summary()['paths_fmi'][0])
    fmi_stat, depth_stat = data_well.get_FMI(key=data_well.well_summary()['paths_fmi'][1])
    print(fmi_dyna.shape, depth_dyna.shape, type(fmi_stat))
    print(fmi_stat.shape, depth_stat.shape, type(fmi_stat))
    # 常规测井数据读取
    data_logging = data_well.get_logging(key='F:\\logging_workspace\\塬22\\塬22_normal_logging_2.csv')
    # 常规测井无效数据归
    data_logging[data_logging<-10000] = 0
    print(data_logging.describe())

    # 图像二值化方法选择
    target_fmi = fmi_dyna
    target_depth = depth_dyna
    # target_method = 'tophat_otsu'
    # target_method = 'otsu'
    # target_method = 'adaptive'
    # target_method = 'kmeans'
    target_method = 'gmm'
    # target_method = 'wavelet'

    # 图像二值化
    result = cal_fmis_segmentation(imgs=[target_fmi], depth=target_depth, windows=400, step=400, method_configs={
        # ('tophat_otsu', 'TopHat + Otsu', None),
        # ('otsu', 'Otsu Threshold', None),
        # ('adaptive', 'Adaptive Threshold', None),
        # ('kmeans', 'K-means (K=3)', None),
        # ('gmm', 'GMM (n=3)', None),
        # ('wavelet', 'Wavelet (db4)', None),
        (target_method, target_method, None)
    })
    for method in list(result.keys()):
        print(method)
        print(result[method][0].shape, type(result[method]))

    # 黄铁矿区域选择，主要需要设置黄铁矿显示的面积阈值
    total_pixels, target_mask = count_small_target_areas(result[list(result.keys())[0]][0], area_threshold=60)

    fmi_data_save(
        save_path=path_well+r"\target_mask.txt",
        img_data=target_mask,
        depth_data=depth_dyna
    )

    # 黄铁矿含量计算
    result_pyrite = cal_fmis_pyrite_content(
            depth_data = target_depth,
            list_fmis = [target_mask],
            config_windows = {'window_length': 200, 'step': 10, 'depth_col_name': 'DEPTH', 'curves_names_list': [target_method+'_fmi']},
    )

    print(result_pyrite.shape, type(result_pyrite))
    print(result_pyrite.describe())

    # data_logging的插值处理
    interpolator = ConventionalLogInterpolator(method='pchip')
    data_logging = interpolator.interpolate_logs(data_logging, depth_col='#DEPTH',target_length=result_pyrite.shape[0])

    # 合并数据，计算的黄铁矿含量 + 其他矿物含量结果
    combined_data = combine_logging_data(
        data_main=result_pyrite,
        data_vice=[data_logging[['#DEPTH', 'GR', 'DWMG_INCP', 'DWFE_INCP', 'DWCA_INCP', 'DWAL_INCP', 'DWSI_INCP', 'ILLITE_QE', 'PYRITE_ORG']]],
        depth_col='DEPTH',
        drop=True
    )
    print(combined_data.describe())

    # LDM = LoggingDataManager(
    #     logging_data=combined_data,
    #     fmi_data={'depth': target_depth, 'image_data': [fmi_dyna, fmi_stat, result[target_method][0], target_mask]},
    # )
    # print('plot depth limits is :', LDM._get_depth_limits())
    #
    # well_viewer = WellLogVisualizer(LDM,
    #     config_logging={'curves_plot' : ['GR', 'DWMG_INCP', 'DWFE_INCP', 'DWCA_INCP', 'DWAL_INCP', 'DWSI_INCP', 'ILLITE_QE', 'PYRITE_ORG']},
    #     # config_type = {'types_cols': 'auto'},
    #     config_fmi = {'color_map': 'hot'}
    #     )
    # config_logging, config_fmi, config_nmr, config_type = well_viewer.get_plot_config()
    # print(config_logging, '\n', config_fmi, '\n', config_nmr, '\n', config_type)
    #
    # well_viewer.visualize()

    # combined_data[['DEPTH', target_method+'_fmi']].to_csv(path_well+'\\'+f'电成像计算黄铁矿—{target_method}.csv',index=False)
    (combined_data[['DEPTH', target_method+'_fmi']]).to_excel(path_well+'\\'+f'电成像计算黄铁矿—{target_method}.xlsx', index=False, sheet_name='0')
    # print(LDM.get_logging_resolution())