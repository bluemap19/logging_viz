import pandas as pd

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_logging.logging_combine import combine_logging_table
from src_logging.logging_to_txt import save_dataframe_to_txt
from src_logging.logging_to_wis import save_dataframe_to_wis
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

    # 常规测井数据读取
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\塬22\\塬22_normal_logging_2.csv')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area80_width20_height20.xlsx')
    data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area120_width20_height20.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area160_width20_height20.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area80_width25_height25.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area120_width25_height25.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area160_width25_height25.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area80_width30_height30.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area120_width30_height30.xlsx')
    # data_logging = data_well.get_logging(key='Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area160_width30_height30.xlsx')

    # 常规测井无效数据归
    data_logging[data_logging<-1000] = 0
    print(data_logging.describe())

    path_core_data = r'Z:\logging_workspace\姬119H2\姬119H2导眼井全岩实验数据_table.csv'

    table_core = pd.read_csv(path_core_data, encoding='GBK')
    # data_core = data_core.rename(columns={"石英": "SY", "斜长石": "XCS", "黄铁矿": "HTK", "黏土矿物": "NTKW"})
    print(table_core.describe())
    print('core table shape is:', table_core.shape)

    # data_all = combine_logging_table(data_main=data_logging, table_vice=table_core, drop=True)
    # print('combined data shape is:', data_all.shape)
    data_all = data_logging

    data_all['GR_scale_factor'] = ((data_all['GR']) - 50)/(400-50)
    data_all['PE_scale_factor'] = ((data_all['PE']) - 2.3)/(4-2.3)
    data_all['Pyrite_GR_adjust_OTSU'] = ((data_all['OTSU_FMI']) * data_all['GR_scale_factor']) + ((data_all['GMM_FMI']) * (1-data_all['GR_scale_factor']))
    data_all['Pyrite_GR_adjust_GMM'] = ((data_all['GMM_FMI']) * data_all['GR_scale_factor']) + ((data_all['OTSU_FMI']) * (1-data_all['GR_scale_factor']))
    data_all['Pyrite_PE_adjust_OTSU'] = ((data_all['OTSU_FMI']) * data_all['PE_scale_factor']) + ((data_all['GMM_FMI']) * (1-data_all['PE_scale_factor']))
    data_all['Pyrite_PE_adjust_GMM'] = ((data_all['GMM_FMI']) * data_all['PE_scale_factor']) + ((data_all['OTSU_FMI']) * (1-data_all['PE_scale_factor']))

    # print(data_all[['#DEPTH', '黄铁矿', 'AC', 'GR', 'DEN', 'GMM_FMI', 'ADAPTIVE_FMI', 'KMEANS_FMI', 'OTSU_FMI', 'TOPHAT_OTSU_FMI', 'WAVELET_FMI', '石英', '钾长石', '斜长石', '黄铁矿', '黏土矿物']].head(10))
    print(data_all.describe())

    # # 相关性分析，计算岩心铁矿物含量与各算法fmi计算结果的相关性
    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df_input=data_all,
    #     target_col='黄铁矿',
    #     # input_cols=['GMM_FMI', 'ADAPTIVE_FMI', 'KMEANS_FMI', 'OTSU_FMI', 'TOPHAT_OTSU_FMI', 'WAVELET_FMI'],
    #     input_cols=['Pyrite_GR_adjust_OTSU', 'Pyrite_GR_adjust_GMM', 'Pyrite_PE_adjust_OTSU', 'Pyrite_PE_adjust_GMM'],
    #     regressor_use=True,
    #     replace_dict={}
    # )


    # data_all[['#DEPTH', 'GR_scale_factor', 'PE_scale_factor','Pyrite_GR_adjust_OTSU', 'Pyrite_GR_adjust_GMM', 'Pyrite_PE_adjust_OTSU', 'Pyrite_PE_adjust_GMM']].to_csv('data_all.csv', index=False)

    # data_logging = data_well.combine_logging_table(
    #     logging_key=r'Z:\\logging_workspace\\姬119H2\\logging+电成像计算黄铁矿_area80_width20_height20.xlsx',
    #     table_key=r'Z:\\logging_workspace\\姬119H2\\姬119H2导眼井全岩实验数据_table_Pyrite.csv'
    # )
    # print(data_logging.head())

