from src_plot.well_logging_viz.data_manager import LoggingDataManager
from src_plot.well_logging_viz.data_visulization import WellLogVisualizer
from src_well_data.data_logging_well import DATA_WELL

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
        nmr_data={'depth': depth_nmr, 'nmr_data': [data_nmr/256]}
    )
    print(LDM._get_depth_limits())

    well_viewer = WellLogVisualizer(LDM,
                                    config_logging={'curves_plot':['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', ['ENG_MEAN_DYNA', 'ENG_SUB_DYNA'], ['HOM_MEAN_DYNA', 'HOM_SUB_DYNA'], ['CON_SUB_DYNA', 'DIS_SUB_DYNA']]},
                                    config_type = {'types_cols': 'auto'},
                                    config_fmi = {'color_map': 'hot'},
                                    config_nmr = {'plot_amplitude_scaling': 40.0, 'x_logarithmic_scale': False}
    )

    config_logging, config_fmi, config_nmr, config_type = well_viewer.get_plot_config()
    print(config_logging, '\n', config_fmi, '\n', config_nmr, '\n', config_type)

    well_viewer.visualize()

    # # print(LDM.get_logging_resolution())