from src_well_data_base.data_logging_table import DataTable


if __name__ == '__main__':
    # 用户提供的测试用例
    test_case = {
            'path': r'F:\logging_workspace\FY1-12\logging_table_result.csv',
            'path_save': r'F:\logging_workspace\FY1-12\logging_table3_result.csv',
            'well_name': 'FY1-12',
        }


    # 创建DataTable实例
    class_table = DataTable(
        path=test_case['path'],
        well_name=test_case['well_name']
    )

    print(class_table.get_table_2())
    print(class_table.get_table_3())

    table_3 = class_table.get_table_3()
    table_3.to_csv(path_or_buf=test_case['path_save'], index=True)