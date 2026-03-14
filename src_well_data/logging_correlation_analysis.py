import pandas as pd

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_data_process.data_dilute import dilute_dataframe
from src_data_process.data_distribution_statistics_overview import data_overview
from src_data_process.data_supervised import supervised_classification, model_predict
from src_data_process.data_supervised_evaluation import evaluate_supervised_clustering
from src_logging.logging_combine import combine_logging_data
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_table.table_process import table_2_to_3
from src_well_data_base.data_logging_well import DATA_WELL

if __name__ == '__main__':
    # well = DATA_WELL(r'F:\logging_workspace\SIMU5-Crack-Hole')
    well = DATA_WELL(r'F:\logging_workspace\樊页2HF')

    summary_temp = well.well_summary()
    for k, val in summary_temp.items():
        print(k, val)

    # logging_data_temp = well.get_logging()
    # print(logging_data_temp.describe())

    path_list_fmi = well.get_path_list_fmi()
    print(path_list_fmi)
    path_list_logging = well.get_path_list_logging()
    print(path_list_logging)
    path_list_table = well.get_path_list_table()
    print(path_list_table)

    replace_dict = well.get_table_replace_dict()
    replace_dict_inversed = {value: key for key, value in replace_dict.items()}
    print(replace_dict, replace_dict_inversed)

    # cols_texture = list(logging_data_type.columns)[1:-1]
    cols_texture = ['ASM_SUB_DYNA', 'HOM_Y_DYNA', 'DIS_Y_DYNA', 'HOM_SUB_DYNA', 'ENT_SUB_DYNA', 'DIS_MEAN_STAT']        # 这个是经过筛选的纹理特征
    print(cols_texture)

    target_col = 'Cracks_Type'
    logging_data_type = well.combine_logging_table(
            logging_key=path_list_logging[0],
            curve_names_logging=cols_texture,
            table_key=path_list_table[0],
            replace_dict={},
            new_col=target_col,
            norm=False,
    )
    print(list(logging_data_type.columns))
    print(logging_data_type.describe())

    # 按组抽稀，每个组保留50%的数据
    logging_data_dilute = dilute_dataframe(logging_data_type, ratio=20, method='random', group_by=target_col)
    print(f"按组抽稀50%后形状: {logging_data_dilute.shape}")
    plot_matrxi_scatter(logging_data_dilute, cols_texture[:6], target_col, target_col_dict={0: '层理缝', 1: '高导缝', 2: '高阻缝', 3: '诱导缝', 4: '断层'}, figsize=(14, 12))

    # 执行分析
    result = data_overview(
        df=logging_data_type,
        input_names=cols_texture,
        target_col=target_col,
        target_col_dict={0: '层理缝', 1: '高导缝', 2: '高阻缝', 3: '诱导缝', 4: '断层'},
    )
    # 保存结果
    result.to_excel('data_overview.xlsx', index=True)

    # df_results, trained_classifiers = supervised_classification(
    #     logging_data_type[cols_texture], logging_data_type[target_col],
    #     Norm=True,  # 不跳过标准化
    #     Type_str={'岩相1':0, '岩相2':1, '岩相3':2, '岩相4':3, '岩相5':4},
    #     y_type_number=5,        # 一共有几类
    # )
    #
    # predictions = model_predict(trained_classifiers, logging_data_type[cols_texture])
    # print(predictions.describe())
    #
    # df_all = pd.concat([logging_data_type, predictions], axis=1)
    # Type_cols = ['Cracks_Type', 'MLP', 'KNN', 'SVM', 'Naive Bayes', 'Random Forest', 'GBM']
    # for col in Type_cols:
    #     df_all[col] = df_all[col].map(replace_dict_inversed)
    #
    #     table_2_temp = df_all[['DEPTH', col]]
    #     table_3_array = table_2_to_3(table_2_temp.values)
    #     # 将转换结果重新封装为DataFrame
    #     table_3_temp = pd.DataFrame(table_3_array, columns=['DEPTH_START', 'DEPTH_END', col])
    #     table_3_temp.to_csv(f'{col}.csv', index=True)
    #
    # df_all.to_csv(well.well_path+'\\result_all.csv', index=False)
    #
    # df_eval = pd.concat([logging_data_type[target_col], predictions], axis=1)
    # print(df_eval.describe())
    # print(df_eval.columns)


    # result = evaluate_supervised_clustering(
    #     df=df_eval,
    #     col_org='Cracks_Type',
    #     cols_compare=['MLP', 'KNN', 'SVM', 'Naive Bayes', 'Random Forest', 'GBM'],
    #     save_report=True,  # 设置为True可保存Excel报告
    #     report_path="test_evaluation.xlsx"
    # )
    # for key, value in result.items():
    #     print(key, result)

    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df_input=logging_data_type,
    #     input_cols=cols_texture,
    #     target_col=target_col,
    #     regressor_use=False,
    #     replace_dict={},
    # )
    # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # print("\n按随机森林特征重要性排序的属性:", rf_sorted)

    # logging_texture = well.get_logging(key=path_list_logging[0])
    # logging_paras = well.get_logging(key=path_list_logging[1])
    #
    # print(logging_paras.describe())
    # print(logging_texture.describe())
    #
    # data_all = combine_logging_data(
    #         data_main = logging_paras,
    #         data_vice = [logging_texture],
    #         depth_col = 'DEPTH',
    #         drop = True
    # )
    # print(data_all.describe())
    #
    # cols_texture = list(logging_texture.columns)
    # cols_paras = list(logging_paras.columns)
    # print(cols_paras)
    # print(cols_texture)
    #
    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df_input=data_all,
    #     input_cols=cols_texture[1:],
    #     target_col='HOLE_DENSITY',
    #     regressor_use=True,
    #     replace_dict={},
    # )
    # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # print("\n按随机森林特征重要性排序的属性:", rf_sorted)

    # cols_choiced_cracks = ['COR_Y_STAT', 'ENT_SUB_STAT', 'CON_Y_STAT', 'ENT_MEAN_DYNA', 'COR_X_DYNA', 'ENT_X_DYNA']
    # cols_choiced_holes = ['HOM_MEAN_DYNA', 'ASM_X_DYNA', 'ENG_X_DYNA', 'ENT_X_STAT', 'DIS_Y_STAT', 'ENT_SUB_STAT']

    # path_logging_target = well.search_logging_path_list(new_kw=['120', 'TEXTURE', 'logging'])
    # print(path_logging_target)
    # # path_table_target = well.search_table_path_list(new_kw=['table'])
    # # print(path_table_target)
    # path_fmi_dyna_target = well.search_fmi_path_list(new_kw=["DYNA"])
    # print(path_fmi_dyna_target)
    # path_fmi_stat_target = well.search_fmi_path_list(new_kw=['STAT'])
    # print(path_fmi_stat_target)

    # # # well.get_FMI_fdes(fde_config={'windows_length': 160, 'windows_step': 40, 'processing_method': 'original'})
    # #
    # # # # texture_dyna = well.get_FMI_texture(key='F:\\logging_workspace\\云安012-X18\\云安012-X18-DYNA.txt', texture_config = {
    # # # #         'level': 16,  # 灰度级别
    # # # #         'distance': [2, 4],  # 像素距离
    # # # #         'angles': [0, np.pi / 2],  # 角度方向
    # # # #         'windows_length': 80,  # 窗口长度
    # # # #         'windows_step': 10  # 滑动步长
    # #
    # # # # })
    # # # # texture_stat = well.get_FMI_texture(key='F:\\logging_workspace\\云安012-X18\\云安012-X18-STAT.txt', texture_config = {
    # # # #         'level': 16,  # 灰度级别
    # # # #         'distance': [2, 4],  # 像素距离
    # # # #         'angles': [0, np.pi / 2],  # 角度方向
    # # # #         'windows_length': 80,  # 窗口长度
    # # # #         'windows_step': 10  # 滑动步长
    # # # # })
    # # # # print(texture_dyna.describe())
    # # # # print(texture_stat.describe())
    # #
    # # texture_all = well.get_FMI_textures(texture_config={
    # #         'level': 16,  # 灰度级别
    # #         'distance': [2, 4],  # 像素距离
    # #         'angles': [0, np.pi / 2],  # 角度方向
    # #         'windows_length': 120,  # 窗口长度
    # #         'windows_step': 10  # 滑动步长
    # # })
    # # print(texture_all.describe())
    # #
    # # # input_cols = ['AC', 'CAL', 'CNL', 'DEN', 'DTS', 'GR', 'RT', 'RXO']
    # # # input_cols = ['CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'COR_MEAN_STAT', 'ASM_MEAN_STAT', 'ENT_MEAN_STAT', 'CON_SUB_STAT', 'DIS_SUB_STAT', 'HOM_SUB_STAT', 'ENG_SUB_STAT', 'COR_SUB_STAT', 'ASM_SUB_STAT', 'ENT_SUB_STAT', 'CON_X_STAT', 'DIS_X_STAT', 'HOM_X_STAT', 'ENG_X_STAT', 'COR_X_STAT', 'ASM_X_STAT', 'ENT_X_STAT', 'CON_Y_STAT', 'DIS_Y_STAT', 'HOM_Y_STAT', 'ENG_Y_STAT', 'COR_Y_STAT', 'ASM_Y_STAT', 'ENT_Y_STAT']
    # # # input_cols = ['COR_MEAN_STAT', 'COR_Y_STAT', 'ASM_SUB_STAT', 'HOM_SUB_STAT', 'COR_X_STAT', 'ENG_SUB_STAT', 'ENT_SUB_STAT', 'HOM_X_STAT']

    # # # input_cols = ['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA', 'COR_SUB_DYNA', 'ASM_SUB_DYNA', 'ENT_SUB_DYNA', 'CON_X_DYNA', 'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA', 'COR_X_DYNA', 'ASM_X_DYNA', 'ENT_X_DYNA', 'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA', 'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA']
    # # # input_cols = ['ENT_SUB_DYNA', 'HOM_X_DYNA', 'ENG_SUB_DYNA', 'HOM_SUB_DYNA', 'ASM_SUB_DYNA', 'CON_SUB_DYNA', 'CON_X_DYNA', 'DIS_SUB_DYNA']
    # input_cols = ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'TEXTURE_LEVEL']
    #
    # # target_col = 'Type'
    # # logging_data_temp_type = well.combine_logging_table(
    # #         logging_key=path_logging_target[0],
    # #         curve_names_logging=input_cols,
    # #         table_key=path_table_target[0],
    # #         replace_dict={},
    # #         new_col=target_col,
    # #         norm=False,
    # # )
    # # print(list(logging_data_temp_type.columns))
    # # print(logging_data_temp_type.describe())
    # logging_data_temp_type = well.get_logging(key=path_logging_target[0], curve_names=input_cols)
    #
    # # 创建模型实例
    # model = MultiVariateLinearRegressor(fit_intercept=True)
    # # # 训练模型（使用x1, x2, x3预测y1, y2）
    # # model.fit(logging_data_temp_type, x_cols=['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT'], y_cols=['Type'])
    # coef_matrix = np.array([-2.82736617, 1.65717694, -1.32970234, -15.03774471]).reshape((1, 4))
    # intercept_matrix = np.array([2.08278412]).reshape((1, 1))
    # print('intercept_matrix is :', intercept_matrix.shape, 'coef_matrix is :', coef_matrix.shape)
    # model.set_fit_paras(x_cols=['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT'], y_cols=['Type'], intercept_matrix=intercept_matrix, coef_matrix=coef_matrix)
    # ############## 进行预测
    # logging_data_temp_type['TEXTURE_LEVEL'] = model.predict(logging_data_temp_type)
    #
    # def classify_texture(value):
    #     if value < 1.0:
    #         return 0
    #     elif 1.0 <= value < 1.3:
    #         return 1
    #     elif 1.5 <= value < 1.78:
    #         return 2
    #     else:
    #         return 3
    # logging_data_temp_type['TYPE_PRED'] = logging_data_temp_type['TEXTURE_LEVEL'].apply(classify_texture)
    #
    # logging_data_temp_type.to_csv(well.well_path+'\\'+well.WELL_NAME+'_result.csv', index=False)
    #
    # # ############## 计算评估指标
    # # test_metrics = model.score(logging_data_temp_type)
    # # print(type(model.coef_matrix), type(model.intercept_matrix), model.coef_matrix.shape, model.intercept_matrix.shape, model.coef_matrix, model.intercept_matrix)
    #
    # # # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    # # #     df_input=logging_data_temp_type,
    # # #     input_cols=input_cols,
    # # #     target_col=target_col,
    # # #     regressor_use=False,
    # # #     replace_dict={},
    # # # )
    # # # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # # # print("\n按随机森林特征重要性排序的属性:", rf_sorted)
    # #
    # # # # 按组抽稀，每个组保留50%的数据
    # # # logging_data_dilute = dilute_dataframe(logging_data_temp_type, ratio=5, method='random', group_by=target_col)
    # # # print(f"按组抽稀50%后形状: {logging_data_dilute.shape}")
    # # # plot_matrxi_scatter(logging_data_dilute, ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'CON_X_DYNA', 'CON_SUB_DYNA', 'COR_MEAN_STAT'], target_col, target_col_dict={})
    #
    # depth_config = [logging_data_temp_type['DEPTH'].min(), logging_data_temp_type['DEPTH'].max()]
    # fmi_dynamic, depth_dyna = well.get_FMI(key=path_fmi_dyna_target[0], depth=depth_config)
    # fmi_static, depth_stat = well.get_FMI(key=path_fmi_stat_target[0], depth=depth_config)
    #
    # # 使用类接口进行可视化
    # print("创建可视化器...")
    # visualizer = WellLogVisualizer()
    # try:
    #     # 启用详细日志级别
    #     logging.getLogger().setLevel(logging.INFO)
    #
    #     # 执行可视化
    #     visualizer.visualize(
    #         logging_dict={'data': logging_data_temp_type,
    #                       'depth_col': 'DEPTH',
    #                       'curve_cols': ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'TEXTURE_LEVEL'],  # 选择显示的曲线
    #                       # 'type_cols': ['Type', 'TYPE_PRED'],  # 分类数据
    #                       'type_cols': ['TYPE_PRED'],  # 分类数据
    #                       'legend_dict': {0: '不发育', 1: '欠发育', 2: '较发育', 3: '发育'}  # 图例定义
    #                       },
    #         fmi_dict={  # FMI图像数据
    #             'depth': depth_dyna,
    #             'image_data': [fmi_dynamic, fmi_static],
    #             'title': ['FMI动态', 'FMI静态']
    #         },
    #         # # NMR_dict=[NMR_DICT1, NMR_DICT2],
    #         # NMR_dict={  # NMR数据
    #         #     'depth': depth_fmi,
    #         #     'nmr_data': [fmi_dynamic, fmi_static],
    #         #     'title': ['NMR动态', 'NMR静态']
    #         # },
    #         # NMR_CONFIG={'X_LOG': [False, True], 'NMR_TITLE': ['N1_谱', 'N2_谱'], 'X_LIMIT': [[1, 1000], [1, 1000]],
    #         #             'Y_scaling_factor': 12, 'JUMP_POINT': 15},
    #         # depth_limit_config=[320, 380],  # 深度限制
    #         figsize=(14, 12)  # 图形尺寸
    #     )
    #
    #     # 显示性能统计
    #     stats = visualizer.get_performance_stats()
    #     print("性能统计:", stats)
    #
    # except Exception as e:
    #     print(f"可视化过程中出现错误: {e}")
    #     import traceback
    #
    #     traceback.print_exc()  # 打印完整错误堆栈
    # finally:
    #     # 清理资源
    #     visualizer.close()