import time
import pandas as pd
import os


def Save_result_as_csv(file_name, dict_results, folder=None):
    """
    Saves a result as a CSV file.

    :param file_name:
    :param dict_results:
    :param folder:
    :return:
    """

    timestr = time.strftime(" %H|%M|%S %d-%m-%Y")
    file_name = file_name + timestr + '.csv'
    dataframe = pd.DataFrame(data=dict_results, columns=dict_results.keys()) #, ignore_index=True)

    if folder== None:
        current_dir = os.getcwd()
        dirPath1 = os.path.join(current_dir, file_name)
        export_csv = dataframe.to_csv(dirPath1, index=None, header=True)

    else:
        current_dir = os.getcwd()
        dir1 = os.path.join(current_dir, folder)
        dirPath2 = os.path.join(dir1, file_name)
        export_csv = dataframe.to_csv(dirPath2, index=None, header=True)





