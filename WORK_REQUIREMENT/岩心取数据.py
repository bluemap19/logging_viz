import pandas as pd

from src_data_process.data_correction_analysis import feature_influence_analysis
from src_well_data.data_logging_well import DATA_WELL

if __name__ == '__main__':
    # path_core_data = r'Z:\logging_workspace\塬22\塬22-岩心矿物含量.csv'
    # path_core_data = r'Z:\logging_workspace\塬22\塬22-岩心矿物含量.csv'
    path_core_data = r'Z:\logging_workspace\姬119H2\姬119H2导眼井全岩实验数据.csv'

    data_core = pd.read_csv(path_core_data, encoding='GBK')
    # data_core = data_core.rename(columns={"石英": "SY", "斜长石": "XCS", "黄铁矿": "HTK", "黏土矿物": "NTKW"})
    print(data_core.describe())
    print(data_core.shape)
    # exit(0)

    # 井设置
    # path_well = r'Z:\logging_workspace\塬22'
    path_well = r'Z:\logging_workspace\姬119H2'
    data_well = DATA_WELL(path_well)
    print(data_well.well_path)
    print(data_well.well_summary())

    # 常规测井数据读取
    data_logging = data_well.get_logging(key=r'Z:\logging_workspace\姬119H2\姬119H2常规测井_logging_data.csv')
    print(data_logging.describe())

    for i in range(data_core.shape[0]):
        depth_point = data_core.iloc[i, 0]
        print(f"处理深度点: {depth_point}")

        data_temp = data_well.get_logging(key=r'Z:\logging_workspace\姬119H2\姬119H2常规测井_logging_data.csv', depth_limit=[depth_point-0.05, depth_point+0.05])
        # print(data_temp.shape)
        data_average = data_temp.mean()

        # ✅ 关键修正：只更新当前行 i 的数据
        for col in data_temp.columns:
            if col in data_core.columns:
                data_core.loc[i, col] = data_average[col]
            else:
                # 如果列不存在，创建新列
                data_core[col] = pd.NA
                data_core.loc[i, col] = data_average[col]

    # # data_core.to_excel(path_well + r'\塬22-岩心总数据.xlsx', index=False, sheet_name='0')
    # data_core.to_excel(path_well + r'\姬119H2-岩心取测井数据.xlsx', index=False, sheet_name='0')
    print(data_core.head())

    # 相关性分析，计算岩心铁矿物含量与各算法fmi计算结果的相关性
    pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
        df_input=data_core,
        target_col='黄铁矿',
        input_cols=['WAVELET_FMI', 'KMEANS_FMI', 'OTSU_FMI', 'ADAPTIVE_FMI', 'TOPHAT_OTSU_FMI', 'GMM_FMI'],
        regressor_use=True,
        replace_dict={}
    )
