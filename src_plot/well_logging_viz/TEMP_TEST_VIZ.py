from src_plot.well_logging_viz.data_manager import LoggingDataManager
from src_plot.well_logging_viz.data_visulization import WellLogVisualizer
from src_well_data.data_logging_well import DATA_WELL

if __name__ == '__main__':
    # work_well = DATA_WELL(path_folder=r'F:\logging_workspace\FY1-15')
    work_well = DATA_WELL(path_folder=r'F:\logging_workspace\桃镇1H')

    summary_dict = work_well.well_summary()
    for well in summary_dict.keys():
        print(well, '\t:\t', summary_dict[well])

    # logging_data = work_well.combine_logging_table(logging_key=path_list_logging[0], table_key=path_list_table[0],
    #                                                curve_names_logging=['DEPTH', 'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA']
    #                                                )
    # print(logging_data.describe())
    # print(logging_data.head(10))
    # COLS_ALL = logging_data.columns.to_list()
    # print(f"总列数: {len(COLS_ALL)}")
    # print(f"所有列名: {COLS_ALL}")
    # print(work_well.get_table_replace_dict())

    logging_data = work_well.get_logging(key=r'F:\\logging_workspace\\桃镇1H\\桃镇1H_normal_logging_data.csv', curve_names=['AC', 'CN', 'DEN', 'GRC'])
    print('logging data total shape is :{}, and its cols including:{}'.format(logging_data.shape, logging_data.columns))

    fmi_dyna, depth_fmi = work_well.get_FMI(key=r'F:\\logging_workspace\\桃镇1H\\桃镇1H_DYNA_target.txt')
    fmi_stat, depth_fmi = work_well.get_FMI(key=r'F:\\logging_workspace\\桃镇1H\\桃镇1H_STAT_target.txt')
    print('current fmi depth is from {} to {}, fmi_dyna and fmi_stat shape is {} and {}'.format(depth_fmi[0], depth_fmi[-1], fmi_dyna.shape, fmi_stat.shape))

    fde_dyna = work_well.get_FMI_fde(r'F:\\logging_workspace\\桃镇1H\\桃镇1H_DYNA_target.txt', fde_config={'windows_length': 200, 'windows_step': 50, 'processing_method': 'original'})
    fde_stat = work_well.get_FMI_fde(r'F:\\logging_workspace\\桃镇1H\\桃镇1H_STAT_target.txt', fde_config={'windows_length': 200, 'windows_step': 50, 'processing_method': 'original'})
    print(fde_dyna.shape, fde_stat.shape)

    # LDM = LoggingDataManager(
    #     logging_data=logging_data,
    #     fmi_data={'depth': depth_fmi, 'image_data': [fmi_dyna, fmi_stat]},
    #     nmr_data={'depth': depth_nmr, 'nmr_data': [data_nmr/256]}
    # )
    # print(LDM._get_depth_limits())

    # # initial_stats = LDM.get_performance_stats()
    # # num_iterations = 3
    # #
    # # import time
    # # # 测试常规测井数据
    # # print("\n📊 测试常规测井数据...")
    # # for i, depth_config in enumerate(depth_config_list):
    # #     top_depth, bottom_depth = depth_config
    # #     times = []
    # #
    # #     start_time = time.perf_counter()
    # #     data = LDM.get_visible_logging_data(top_depth, bottom_depth)
    # #     # print(data.describe())
    # #     data = LDM.get_visible_fmi_data(top_depth, bottom_depth)
    # #     if data is not None:
    # #         print(data['title'])
    # #         print(data['image_data'][0].shape)
    # #         print(data['depth'].shape)
    # #     else:
    # #         print(data)
    # #     elapsed = (time.perf_counter() - start_time) * 1000
    # #     print(f"  范围 [{top_depth}-{bottom_depth}]: 平均 {elapsed:.2f}ms")
    #
    # well_viewer = WellLogVisualizer(LDM,
    #                                 config_logging={'curves_plot':['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', ['ENG_MEAN_DYNA', 'COR_MEAN_DYNA'], ['ASM_MEAN_DYNA', 'ENT_MEAN_DYNA'], ['CON_SUB_DYNA', 'DIS_SUB_DYNA']]},
    #                                 # config_type = {'types_cols': 'auto'},
    #                                 # config_fmi = {'color_map': 'hot'}
    #                                 )
    # config_logging, config_fmi, config_nmr, config_type = well_viewer.get_plot_config()
    # print(config_logging, '\n', config_fmi, '\n', config_nmr, '\n', config_type)
    #
    # well_viewer.visualize()
    #
    # # print(LDM.get_logging_resolution())