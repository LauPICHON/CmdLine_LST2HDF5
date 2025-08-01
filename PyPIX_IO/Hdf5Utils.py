import h5py
import random
import os
import numpy as np
import threading
from datetime import datetime
from time import perf_counter

# Static Class
class Hdf5Handler():

    _FILE_LOCK_1 = threading.Lock()

    @staticmethod
    def save_hdf5(filename, points_map, spectrums_list, group_name="XRF_Analysis", dataset_name="dataset"):
        try :
            assert(len(points_map) == len(spectrums_list))
        except AssertionError:
            raise(IndentationError(f"points_map length : {len(points_map)} and spectrum_list length : {len(spectrums_list)} do not match."))
        
        with h5py.File(filename, 'w') as h5file:
            
            data = np.zeros((3, 3, 3, len(spectrums_list[0])))

            for index, point in enumerate(points_map) :
                x = point[0]
                y = point[1]
                z = point[2]
                data[x,y,z] = spectrums_list[index]
            
            subgroup = h5file.require_group(group_name)
            subgroup.attrs['Analysis date'] = datetime.now().strftime('%d/%m/%Y')
            subgroup.attrs['Analysis time'] = datetime.now().strftime("%H:%M:%S")
            dset = subgroup.create_dataset(dataset_name, data=data)

    @staticmethod
    def create_empty_hdf5(filename: str, data_shape, dtype=np.float64, group_name="XRF_analysis", dataset_name="dataset", file_lock=_FILE_LOCK_1):
        with file_lock :
            with h5py.File(filename, 'w') as h5file:
                subgroup = h5file.require_group(group_name)
                dset = subgroup.create_dataset(dataset_name, shape=data_shape, dtype=dtype)

    @staticmethod
    def feed_existing_hdf5(filename, data, group_name="XRF_analysis", dataset_name="dataset", file_lock=_FILE_LOCK_1):
        
        with file_lock :
            with h5py.File(filename, 'a') as h5file:
                dset = h5file[f'{group_name}/{dataset_name}']
                dset[:] = data

    @staticmethod
    def get_dataset_data_hdf5(filename, group_name:str="XRF_analysis", dataset_name:str="dataset", file_lock=_FILE_LOCK_1):
        
        with file_lock :
            with h5py.File(filename, 'r') as h5file:
                group = h5file.require_group(f'{group_name}')
                dset_data = group[f'{dataset_name}']
                np_dset_data = np.array(dset_data)

            return np_dset_data

    @staticmethod
    def feed_spectrum(filename, spectrum:list[int], x_position:int, y_position:int, group_name="XRF_analysis", dataset_name="dataset", file_lock=_FILE_LOCK_1) -> None :
        try :
            with file_lock : 
                with h5py.File(filename, 'a') as h5file:
                    group = h5file.require_group(f'{group_name}')
                    dset = group[f'{dataset_name}']
                    dset[y_position, x_position] = spectrum
        except IndexError as idxerr :
            print(f'INDEXERROR : {x_position} or {y_position} out of range \n {idxerr}')
            raise IndexError(idxerr)

    @staticmethod
    def save_final_hdf5_from_tmp(save_filepath, tmp_file, channels, calibration, live_time, tmp_group_name="XRF_analysis", tmp_dataset_name="dataset"):
        with h5py.File(tmp_file, 'r') as tmp_file:
            group = tmp_file.require_group(f'{tmp_group_name}')
            dset_data = group[f'{tmp_dataset_name}']
            np_dset_data = np.array(dset_data)
        
        with h5py.File(save_filepath, 'w') as final_hdf5:
            mca0 = final_hdf5.require_group("mca_0")
            data = mca0.create_dataset("data", data=np_dset_data)
            chan = mca0.create_dataset("channels", data=channels)
            calib = mca0.create_dataset("calibration", data=calibration)
            ltime = mca0.create_dataset("live_time", data=live_time)

if __name__ == "__main__":

    # x = [0,1,2]
    # y = [0,1,2]
    # z = [0,1,2]
    # points_map = [[0,0,0], [0,1,0], [0,2,0], [1,2,0], [1,1,0], [1,0,0], [2,0,0], [2,1,0], [2,2,0]]

    # spectrums = [[random.randint(100,1000) for index in range(255)] for index in range(9)]
    # spectrum = [range(512)]


    # Hdf5Handler.save_hdf5("test.hdf5", points_map, spectrums)
    dirname = os.path.dirname(os.path.abspath(__file__))
    
    file = os.path.join(dirname, "myNewHdf5.hdf5")
    # print(file)
    Hdf5Handler.create_empty_hdf5(file, (50,3,3,512))
    import time
    data = np.ones((50,3,3,512))
    for index in range(30):
        data[index][0][0] = index  * np.ones(512)
        Hdf5Handler.feed_existing_hdf5(file, data)
        # print("loop")
        time.sleep(1)



