def data_preprocessing(origin_csv_path, dest_csv_path):
    # 1. Load dataset
    # 2. Data preprocessing
    # 3. Save processed data

    from datetime import datetime
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # Load dataset CSV
    df = pd.read_csv(origin_csv_path, sep=';', thousands='.',decimal=',')

    # Drop redundant columns
    cols_drop = ['Customer_ID','infobase','actvsubs','adjmou','adjqty','adjrev',
            'avg3mou','avg3qty','avg3rev','avg6mou','avg6qty','avg6rev','avgrev',
            'blck_dat_Mean','blck_vce_Mean','ccrndmou_Mean','comp_dat_Mean',
            'comp_vce_Mean','custcare_Mean','drop_dat_Mean','drop_vce_Mean',
            'opk_dat_Mean','opk_vce_Mean','ovrmou_Mean','owylis_vce_Mean',
            'peak_dat_Mean','peak_vce_Mean','totcalls','totmrc_Mean',
            'totrev']

    df.drop(columns=cols_drop,inplace=True)

    # Create new features
    df['mou_comp_Mean'] = df['mou_cdat_Mean'] + df['mou_cvce_Mean']
    df.drop(columns=['mou_cdat_Mean','mou_cvce_Mean'], inplace=True)

    df['mou_opk_Mean'] = df['mou_opkd_Mean'] + df['mou_opkv_Mean']
    df.drop(columns=['mou_opkd_Mean','mou_opkv_Mean'], inplace=True)

    df['mou_pea_Mean'] = df['mou_pead_Mean'] + df['mou_peav_Mean']
    df.drop(columns=['mou_pead_Mean','mou_peav_Mean'], inplace=True)

    df['mou_w2w_Mean'] = df['mouiwylisv_Mean'] + df['mouowylisv_Mean']
    df.drop(columns=['mouiwylisv_Mean','mouowylisv_Mean'], inplace=True)

    df['plcd_Mean'] = df['plcd_dat_Mean'] + df['plcd_vce_Mean']
    df.drop(columns=['plcd_dat_Mean','plcd_vce_Mean'], inplace=True)

    # Keep only columns with less than 30% missing values
    mask = df.isna().sum() / len(df) < 0.3
    df = df.loc[:, mask]

    # Drop rows with NAs
    df = df.dropna()

    # List of columns for Label Encoding
    label_encoded=['ethnic','crclscod','area','asl_flag','refurb_new','hnd_webcap','kid0_2','kid3_5','kid6_10','kid11_15',
                'kid16_17','creditcd']

    #Label Encoding for object to numeric conversion
    le = LabelEncoder()

    for feat in label_encoded:
        df[feat] = le.fit_transform(df[feat].astype(str))

    # One Hot Encoding for the remaining categorical features
    # encode categorical variables
    df = pd.get_dummies(df, prefix_sep='_')

    print(f'Dataset shape: {df.shape}')

    df.to_csv(dest_csv_path,index=False)