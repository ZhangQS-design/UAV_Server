import xlwt
import pandas as pd

def data_write(file_path, datas):

    dataframe = pd.DataFrame(datas)
    dataframe.to_excel(file_path)
