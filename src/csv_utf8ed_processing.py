import pandas as pd

def processing(filename, col_name_orig, new_col_name, all_numeric=True):
    df1 = pd.read_csv(filename)
    df2 = df1[['GEO.id2', col_name_orig]]
    df2 = df2[df2['GEO.id2'] != 'Id2']
    df2.columns = ['geo_id', new_col_name]
    if all_numeric: df2[new_col_name] = df2[new_col_name].astype(int)
    return(df2)

if __name__ == '__main__':
    main()
