import time
import pandas as pd
import os
import json


def Save_result_as_csv(file_name, dict_results, dict_parameters, folder=None):
    """
    Saves a result as a CSV file.

    :param file_name:
    :param dict_results:
    :param folder:
    :return:
    """

    timestr = time.strftime(" %H|%M|%S %d-%m-%Y")
    file_name_csv = file_name + timestr + '.csv'
    file_name_json = file_name + timestr + '.json'
    dataframe = pd.DataFrame(data=dict_results, columns=dict_results.keys()) #, ignore_index=True)

    if folder is None:
        current_dir = os.getcwd()
        dirPath_csv = os.path.join(current_dir, file_name_csv)

        #export
        dataframe.to_csv(dirPath_csv, index=None, header=True)

        with open(os.path.join(current_dir, file_name_json), "w") as f_handle:
            json.dump(dict_parameters, f_handle)



    else:
        current_dir = os.getcwd()
        dir1 = os.path.join(current_dir, folder)
        dirPath_csv = os.path.join(dir1, file_name_csv)

        #export
        dataframe.to_csv(dirPath_csv, index=None, header=True)

        with open(os.path.join(dir1, file_name_json), "w") as f_handle:
            json.dump(dict_parameters, f_handle)

