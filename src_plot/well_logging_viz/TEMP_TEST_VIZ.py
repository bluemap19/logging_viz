from src_plot.well_logging_viz.data_manager import LoggingDataManager
from src_plot.well_logging_viz.data_visulization import WellLogVisualizer
from src_well_data.data_logging_well import DATA_WELL
import numpy as np

if __name__ == '__main__':
    # work_well = DATA_WELL(path_folder=r'F:\logging_workspace\FY1-15')
    work_well = DATA_WELL(path_folder=r'Z:\logging_workspace\桃镇1H')

    summary_dict = work_well.well_summary()
    for well in summary_dict.keys():
        print(well, '\t:\t', summary_dict[well])

    path_list_logging = summary_dict['paths_logging']
    path_list_fmi = summary_dict['paths_fmi']
    path_list_table = summary_dict['paths_table']
    path_list_nmr = summary_dict['paths_nmr']

    # 取np.ndarray的图像数据数组
    fmi_dyna, depth_fmi = work_well.get_FMI(key='Z:\\logging_workspace\\桃镇1H\\桃镇1H_DYNA_FULL.txt')
    fmi_stat, depth_fmi = work_well.get_FMI(key='Z:\\logging_workspace\\桃镇1H\\桃镇1H_STAT_FULL.txt')
    # 取pd.Dataframe的data_logging数据数组
    data_logging = work_well.combine_logging_table(
        logging_key='Z:\\logging_workspace\\桃镇1H\\桃镇1H_DYNA_FULL_texture_logging.csv',
        table_key='Z:\\logging_workspace\\桃镇1H\\桃镇1H__LITHO_TYPE.csv',
        curve_names_logging=['DEPTH', 'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA']
    )

    # ========== 生成模拟岩心实验数据（稀疏 DataFrame，大部分为 NaN）==========
    # 岩心数据特点：仅在特定取心深度有数据，其余为 NaN
    n_depths = len(data_logging)
    np.random.seed(42)

    # CORE_GR：模拟伽马岩心数据（仅 5% 深度有值）
    core_gr = np.full(n_depths, np.nan)
    core_gr_indices = np.random.choice(n_depths, size=int(n_depths * 0.05), replace=False)
    core_gr[core_gr_indices] = np.random.uniform(30, 150, size=len(core_gr_indices))

    # CORE_RT：模拟电阻率岩心数据（仅 3% 深度有值）
    core_rt = np.full(n_depths, np.nan)
    core_rt_indices = np.random.choice(n_depths, size=int(n_depths * 0.03), replace=False)
    core_rt[core_rt_indices] = np.random.uniform(1.0, 500.0, size=len(core_rt_indices))

    # CORE_DEN：模拟密度岩心数据（仅 2% 深度有值）
    core_den = np.full(n_depths, np.nan)
    core_den_indices = np.random.choice(n_depths, size=int(n_depths * 0.02), replace=False)
    core_den[core_den_indices] = np.random.uniform(2.0, 2.9, size=len(core_den_indices))

    # 添加到 logging_data DataFrame
    data_logging['CORE_GR'] = core_gr
    data_logging['CORE_RT'] = core_rt
    data_logging['CORE_DEN'] = core_den
    print(f"[岩心数据] CORE_GR 有效点数: {np.sum(~np.isnan(core_gr))}, "
          f"CORE_RT 有效点数: {np.sum(~np.isnan(core_rt))}, "
          f"CORE_DEN 有效点数: {np.sum(~np.isnan(core_den))}")



    # 取np.ndarray的data_nmr图像数据数组
    data_nmr, depth_nmr = work_well.get_NMR(
        key='Z:\\logging_workspace\\桃镇1H\\桃镇1H_DYNA_FULL_fde_NMR.txt'
    )
    print(data_logging.describe())

    print('fmi_dyna shape is :', fmi_dyna.shape, ', data_nmr shape is :', data_nmr.shape, ', data_logging shape is :', data_logging.shape)
    print(data_logging.describe())

    # logging_data = work_well.combine_logging_table(logging_key=path_list_logging[0], table_key=path_list_table[0],
    #                                                curve_names_logging=['DEPTH', 'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA']
    #                                                )
    # print(logging_data.describe())
    # print(logging_data.head(10))
    # COLS_ALL = logging_data.columns.to_list()
    # print(f"总列数: {len(COLS_ALL)}")
    # print(f"所有列名: {COLS_ALL}")
    # print(work_well.get_table_replace_dict())

    # logging_data = work_well.get_logging(key=r'F:\\logging_workspace\\桃镇1H\\桃镇1H_normal_logging_data.csv', curve_names=['AC', 'CN', 'DEN', 'GRC'])
    # print('logging data total shape is :{}, and its cols including:{}'.format(logging_data.shape, logging_data.columns))

    # fde_dyna = work_well.get_FMI_fde(r'F:\\logging_workspace\\桃镇1H\\桃镇1H_DYNA_target.txt', fde_config={'windows_length': 200, 'windows_step': 50, 'processing_method': 'original'})
    # fde_stat = work_well.get_FMI_fde(r'F:\\logging_workspace\\桃镇1H\\桃镇1H_STAT_target.txt', fde_config={'windows_length': 200, 'windows_step': 50, 'processing_method': 'original'})
    # print(fde_dyna.shape, fde_stat.shape)

    LDM = LoggingDataManager(
        logging_data=data_logging,
        fmi_data={'depth': depth_fmi, 'image_data': [255-fmi_dyna, 255-fmi_stat]},
        nmr_data={'depth': depth_nmr, 'nmr_data': [data_nmr/256]},
    )
    print(LDM._get_depth_limits())

    # # ========== 岩心配置 ==========
    # config_core = {
    #     'core_curves': ['CORE_GR', 'CORE_RT', 'CORE_DEN'],  # 3 个岩心曲线
    #     'plot_index_list': [0, 0, 1],  # CORE_GR+CORE_RT 叠加在第0道，CORE_DEN 叠加在第1道
    #     'thicknesses_config': [2.0, 2.5, 1.8],  # 杆粗细
    #     'colors_config': ['#FF6347', '#4169E1', '#32CD32'],  # 橙红、蓝、蓝绿
    #     'alphas_config': [0.6, 0.6, 0.6],  # 透明度（0.0~1.0）
    #     'axis_config': [False, False, False],  # 不使用对数轴
    #     'range_config': [
    #         [0, 150],  # CORE_GR 范围
    #         [0.5, 500],  # CORE_RT 范围
    #         [1.8, 3.0]  # CORE_DEN 范围
    #     ]
    # }
    config_core = {}

    well_viewer = WellLogVisualizer(LDM,
                                    config_logging={'curves_plot':['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', ['ENG_MEAN_DYNA', 'ENG_SUB_DYNA'], ['HOM_MEAN_DYNA', 'HOM_SUB_DYNA'], ['CON_SUB_DYNA', 'DIS_SUB_DYNA']]},
                                    config_type = {'types_cols': 'auto'},
                                    config_fmi = {'color_map': 'hot'},
                                    config_nmr = {'plot_amplitude_scaling': 40.0, 'x_logarithmic_scale': False},
                                    config_core=config_core  # 传入岩心配置
    )

    config_logging, config_fmi, config_nmr, config_type, config_core = well_viewer.get_plot_config()
    print(config_logging, '\n', config_fmi, '\n', config_nmr, '\n', config_type, '\n', config_core)

    well_viewer.visualize()

    # # print(LDM.get_logging_resolution())