import os
import pandas as pd
import numpy as np

EXCEL_FILE = 'Total.xlsx'
OUTPUT_FILE = '金融韧性综合指数_最终版.csv'

EXPECTED_COLUMNS = [
    '省份',
    '不良贷款率',
    '保险深度',
    '外贸依存度',
    '金融机构各项存款余额',
    '金融机构各项贷款余额',
    '社会融资规模增量',
    '上市公司数量',
    '数字普惠金融指数',
    '存贷比',
    '人均GDP',
    '第三产业占GDP比重',
    '金融科技专利授权量',
    '耦合协调度',
]

# 修改点1：将“外贸依存度”归入负向指标
NEGATIVE_COLUMNS = ['不良贷款率', '外贸依存度']
POSITIVE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in ['省份'] + NEGATIVE_COLUMNS]
INDICATOR_COLUMNS = POSITIVE_COLUMNS + NEGATIVE_COLUMNS

# 修改点2：删除了无用的 COLUMN_RENAMES 字典，简化清洗逻辑
def normalize_header(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    # 清除各种不可见字符和空格
    return text.replace('\xa0', '').replace(' ', '')


def read_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine='openpyxl')
    if df.shape[0] < 3:
        raise ValueError(f'工作表 {sheet_name} 行数不足，无法读取指标数据。')

    # 读取第二行作为真实的指标列名
    header = [normalize_header(x) for x in df.iloc[1].tolist()]
    header[0] = '省份'
    df = df.iloc[2:].copy()
    df.columns = header

    # 检查是否有遗漏的列
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f'工作表 {sheet_name} 缺少列: {missing_cols}')

    # 提取所需列并清理省份名称
    df = df[EXPECTED_COLUMNS].copy()
    df['省份'] = df['省份'].astype(str).str.strip()
    df = df[df['省份'].notna()]
    df = df[~df['省份'].isin(['', 'nan', 'None'])]

    # 将指标列转换为数值型
    for col in POSITIVE_COLUMNS + NEGATIVE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 缺失值使用均值填充
    df[POSITIVE_COLUMNS + NEGATIVE_COLUMNS] = df[POSITIVE_COLUMNS + NEGATIVE_COLUMNS].fillna(
        df[POSITIVE_COLUMNS + NEGATIVE_COLUMNS].mean()
    )
    return df


def standardize_data(df):
    df_std = pd.DataFrame({'省份': df['省份']})

    # 正向指标：最大值标准化
    for col in POSITIVE_COLUMNS:
        max_value = df[col].max()
        df_std[col] = df[col] / max_value if max_value != 0 else 0.0

    # 负向指标：倒数转换后最大值标准化
    for col in NEGATIVE_COLUMNS:
        series = df[col].astype(float).copy()
        # 防止出现0或负数导致无法取倒数
        if (series <= 0).any():
            series = series - series.min() + 1e-6
        # 取倒数 (加一个极小值防止分母为0)
        inv_series = 1.0 / (series + 1e-9)
        max_value = inv_series.max()
        df_std[col] = inv_series / max_value if max_value != 0 else 0.0

    return df_std


def calculate_entropy_weights(df_std):
    n = len(df_std)
    if n <= 1:
        return {col: 1.0 / len(INDICATOR_COLUMNS) for col in INDICATOR_COLUMNS}

    k = 1.0 / np.log(n)
    weights = {}

    for col in INDICATOR_COLUMNS:
        series = df_std[col].astype(float)
        total = series.sum()
        if total == 0:
            p = np.full(n, 1.0 / n)
        else:
            p = (series / total).to_numpy(dtype=float)
            # 防止对数计算时出现 log(0)
            p[p <= 0] = 1e-9

        entropy = -k * np.sum(p * np.log(p))
        weights[col] = 1 - entropy

    total_weight = sum(weights.values())
    if total_weight == 0:
        return {col: 1.0 / len(INDICATOR_COLUMNS) for col in INDICATOR_COLUMNS}

    return {col: weight / total_weight for col, weight in weights.items()}


def calculate_composite_index(df_std, weights):
    df_result = pd.DataFrame({'省份': df_std['省份']})
    df_result['综合指数'] = 0.0
    for col in INDICATOR_COLUMNS:
        df_result['综合指数'] += df_std[col] * weights[col]
    return df_result


def sort_sheet_names(sheet_names):
    # 尝试按年份数字排序工作表
    def sheet_key(name):
        try:
            return int(name)
        except ValueError:
            return name
    return sorted(sheet_names, key=sheet_key)


def main():
    if not os.path.exists(EXCEL_FILE):
        print(f" 找不到文件: {EXCEL_FILE}")
        return

    excel_data = pd.ExcelFile(EXCEL_FILE, engine='openpyxl')
    sheet_names = sort_sheet_names(excel_data.sheet_names)
    print(f"读取到工作表: {sheet_names}")

    all_years_results = {}
    for sheet in sheet_names:
        print(f"正在处理工作表: {sheet}")
        df_clean = read_sheet(EXCEL_FILE, sheet)
        df_std = standardize_data(df_clean)
        weights = calculate_entropy_weights(df_std)
        df_index = calculate_composite_index(df_std, weights)
        all_years_results[sheet] = df_index.set_index('省份')['综合指数']

    # 合并所有年份的数据生成面板数据 (Panel Data)
    final_panel_data = pd.DataFrame(all_years_results).reset_index()
    final_panel_data = final_panel_data.dropna(thresh=2)
    final_panel_data.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n🎉 结果已保存为: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()