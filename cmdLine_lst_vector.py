import h5py, re
import numpy as np
import sys, os
import shutil
import threading
from datetime import datetime
from time import perf_counter
import configparser
from time import sleep
import fnmatch
from PIL import Image
#import EdfFile # Import EdfFile for EDF file handling
from PyPIX_IO import EDFStack # Import EDFStack for EDF stack handling

class AGLAEfunction(object):
    _FILE_LOCK_1 = threading.Lock()

    def __init__(self):
        self.path_config_file = "config.ini"
        self.detector = "LE0"
        self.config_adc_all = {}

    @staticmethod
    def ibil_To_hdf5(path_edf, detname, path_lst, adc_metadata_arr):
        """Write data to HDF5 file from a .edf file."""
        metadata_one_adc= adc_metadata_arr[12]  # Assuming 12 is the index for IBIL metadata
        #edfout = EdfFile(edfpath)
        stack = EDFStack.EDFStack()
        stack.loadIndexedStack(path_edf)
        AGLAEfunction.feed_hdf5_map(stack.data, path_lst, "IBIL", 0, metadata_one_adc)


    @staticmethod
    def save_hdf5_antoine(filename, points_map, spectrums_list, group_name="XRF_Analysis", dataset_name="dataset"):
        try:
            assert (len(points_map) == len(spectrums_list))
        except AssertionError:
            raise (IndentationError(
                f"points_map length : {len(points_map)} and spectrum_list length : {len(spectrums_list)} do not match."))

        with h5py.File(filename, 'w') as h5file:

            data = np.zeros((3, 3, 3, len(spectrums_list[0])))

            for index, point in enumerate(points_map):
                x = point[0]
                y = point[1]
                z = point[2]
                data[x, y, z] = spectrums_list[index]

            subgroup = h5file.require_group(group_name)
            subgroup.attrs['Analysis date'] = datetime.now().strftime('%d/%m/%Y')
            subgroup.attrs['Analysis time'] = datetime.now().strftime("%H:%M:%S")
            dset = subgroup.create_dataset(dataset_name, data=data)

    @staticmethod
    def create_empty_prj_hdf5(filename: str, group_name="Global", file_lock=_FILE_LOCK_1):
        '''Create an empty HDF5 file with a group'''
        with file_lock:
            with h5py.File(filename, 'a') as h5file:
                subgroup = h5file.require_group(group_name)

    @staticmethod
    def metadata_prj_hdf5(filename: str, group_name:str, dict_glob_metadata:dict, file_lock=_FILE_LOCK_1):
        '''Create an empty HDF5 file with a group'''
        with file_lock:
            with h5py.File(filename, 'a') as h5file:
                grp = h5file.require_group(group_name)
                for exp_attr in dict_glob_metadata:
                     grp.attrs[exp_attr] = dict_glob_metadata[exp_attr]
                     
                #grp.attrs["Date"] = datetime.now().strftime('%d/%m/%Y')
    

    @staticmethod
    def create_empty_hdf5(filename: str, data_shape, dtype=np.float32, group_name="20250127_0001",
                          dataset_name="dataset", file_lock=_FILE_LOCK_1):
        '''Create an empty HDF5 file with a group and a dataset'''
        with file_lock:
            with h5py.File(filename, 'a') as h5file:
                subgroup = h5file.require_group(group_name)
                dset = subgroup.create_dataset(dataset_name, shape=data_shape, dtype=dtype)

    @staticmethod
    def feed_image_video_hdf5(img_name:str,filename, path_image_dir:str,path_image_video:str,group_name="Screen_capture", file_lock=_FILE_LOCK_1):
        '''Feed an HDF5 file with the video image'''
        with file_lock: 
            with h5py.File(filename, 'a') as h5file:
                group = h5file.require_group(group_name)
                path_img_to_read = os.path.join(path_image_dir, path_image_video[0])
                raw_img = Image.open(path_img_to_read)  # Conversion PIL -> NumPy
                resized_image = raw_img.resize((int(raw_img.width/2),int(raw_img.height/2)))
                rgb_image = resized_image.convert("RGB")
                img = np.array(rgb_image)
                #dataset_name = path_image_video[0].split(".")[0]
                dataset_name = img_name
                try:
                    dataset = group.create_dataset(dataset_name, data=img, compression="gzip", compression_opts=9)
                except:
                    del group[dataset_name]
                    dataset = group.create_dataset(dataset_name, data=img, compression="gzip", compression_opts=9)

                dataset.attrs["CLASS"] = np.bytes_("IMAGE")           # Type de dataset
                dataset.attrs["IMAGE_VERSION"] = np.bytes_("1.2")     # Version du format
                dataset.attrs["IMAGE_SUBCLASS"] = np.bytes_("IMAGE_TRUECOLOR")  # Type de couleur

    @staticmethod
    def feed_image_hdf5(filename, image_path, path_image_dir:str, group_name="Screen_capture", file_lock=_FILE_LOCK_1):
        '''Feed an HDF5 file with an image'''
        with file_lock: 
            with h5py.File(filename, 'a') as h5file:
                group = h5file.require_group(group_name)
            
                try:
                    subgroup = group.create_group("screen_capture")
                except:
                    subgroup = group.get("screen_capture")

                for img1 in image_path:
                    dataset_name = img1.split(".")[0]
                    path_img_to_read = os.path.join(path_image_dir, img1)
                    raw_img = Image.open(path_img_to_read)  # Conversion PIL -> NumPy
                    #resized_image = raw_img.resize((int(raw_img.width/2),int(raw_img.height/2)))
                    rgb_image = raw_img.convert("L")
                    img = np.array(rgb_image)  # Conversion PIL -> NumPy

                    try:
                        dataset = subgroup.create_dataset(dataset_name, data=img, compression="gzip", compression_opts=9)
                    except:
                        del subgroup[dataset_name]
                        dataset = subgroup.create_dataset(dataset_name, data=img, compression="gzip", compression_opts=9)
                    # Métadonnées obligatoires pour HDFView 
                    dataset.attrs["CLASS"] = np.bytes_("IMAGE")           # Type de dataset
                    dataset.attrs["IMAGE_VERSION"] = np.bytes_("1.2")     # Version du format
                    dataset.attrs["IMAGE_SUBCLASS"] = np.bytes_("IMAGE_GRAYSCALE")  # Type de couleur
                    #dataset.attrs["IMAGE_MINMAXRANGE"] = np.array([0, 255], dtype=np.uint8)  # Gamme de valeurs

    @staticmethod
    def feed_existing_hdf5(filename, data, group_name="20250101_0001_STD", dataset_name="dataset", metadata_one_adc=dict, file_lock=_FILE_LOCK_1):
        '''Feed an existing HDF5 file with data'''
        with file_lock:
            with h5py.File(filename, 'a') as h5file:
                subgroup = h5file.require_group(group_name)
                try:
                    dset = subgroup.create_dataset(dataset_name, shape=np.shape(data), dtype=np.int32)
                    for exp_attr in metadata_one_adc:
                        dset.attrs[exp_attr] = metadata_one_adc[exp_attr]
                except:
                    del subgroup[dataset_name]
                    dset = subgroup.create_dataset(dataset_name, shape=np.shape(data), dtype=np.int32)
                    for exp_attr in metadata_one_adc:
                        dset.attrs[exp_attr] = metadata_one_adc[exp_attr]

                dset = h5file[f'{group_name}/{dataset_name}']
                dset[:] = data

    @staticmethod
    def get_dataset_data_hdf5(filename, group_name: str = "XRF_analysis", dataset_name: str = "dataset",
                              file_lock=_FILE_LOCK_1):
        '''Get data from an HDF5 file'''
        with file_lock:
            with h5py.File(filename, 'r') as h5file:
                group = h5file.require_group(f'{group_name}')
                dset_data = group[f'{dataset_name}']
                np_dset_data = np.array(dset_data)

            return np_dset_data
   
    @staticmethod
    def save_final_hdf5_from_tmp(save_filepath, tmp_file, channels, calibration, live_time,
                                 tmp_group_name="XRF_analysis", tmp_dataset_name="dataset"):
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

    @staticmethod
    def write_hdf5(mydata, Pathlst, detector,FinalHDF,num_det):
        destfile = Pathlst.split(".")
        newdestfile = destfile[0] + ".hdf5"

        if destfile[1] == 'lst':
            newdestfile = destfile[0] + ".hdf5"
        elif destfile[1] == 'edf':
            newdestfile = os.path.join(os.path.dirname(Pathlst), FinalHDF)
     
        print(newdestfile)
        hdf = h5py.File(newdestfile, 'a')
        try:
            del hdf[detector]
        except:
            pass
        shape= mydata.shape
        dtype = mydata.dtype
        entryName = 'data' + str(num_det) + "_" + detector  # "data"
        nxData = hdf.require_group(entryName)
        dset = nxData.require_dataset('maps', data = mydata, shape=shape, dtype=dtype, compression="gzip", compression_opts=4)
        nxData.attrs["signal"] = detector
        dset.flush()
        print("HDF5 write" + detector )
        hdf.close()

    @staticmethod
    def feed_hdf5_map(mydata, Pathlst, detector, num_scan_y, dict_metadata_one_adc):
        if mydata.size == 0:
            print(f"No data to write for detector {detector}")
            return
 
        destfile = Pathlst.split(".")
        newdestfile = destfile[0] + ".hdf5"
    
        with h5py.File(newdestfile, 'a') as h5file:
            #group_name = 'data' + str(num_det) + "_" + detector  # "data"
            group_name = detector  # "data"

            if num_scan_y != 0:
                    nxData = h5file[f'{group_name}/maps']
                    nxData.resize((nxData.shape[0] + mydata.shape[0],nxData.shape[1] ,nxData.shape[2]))
                    nxData[-mydata.shape[0]:,0:, :] = mydata
            else:
                    try:
                        del h5file[f'{group_name}']
                    except Exception:
                        pass
                    nxData = h5file.require_group(f'{group_name}')
                    dset = nxData.require_dataset('maps', data = mydata, shape =mydata.shape, dtype=np.uint32, maxshape=(None,None,None), chunks=True, compression="gzip",compression_opts=4)
                    for exp_attr in dict_metadata_one_adc:
                        dset.attrs[exp_attr] = dict_metadata_one_adc[exp_attr]
                        
                    #dset = h5file.require_dataset(group_name, data = mydata, shape =mydata.shape, dtype=np.uint32, maxshape=(None,None,None), chunks=True, compression="gzip",compression_opts=4)
        h5file.close()

    @staticmethod
    def search_ibil_folder(_path_ascii:str):
        """Search for the IBIL folder in the given path."""
        head_tail = os.path.split(_path_ascii)
        root_name = str.split(head_tail[1], sep='_')
        edf_folder = root_name[0] + "_" + root_name[1] + "_EDF"
        
        _path_ibil = os.path.join(head_tail[0], "IBIL")
        ibil_folder_founded = False
        if not os.path.exists(_path_ibil):
            print(f"Path does not exist: {_path_ibil}")
            return False,"none"
        else:
            print(f"Searching for IBIL folder in: {_path_ibil}")

            for root, dirs,files in os.walk(_path_ibil):
                for dir_name in dirs:
                    list_files_edf = fnmatch.filter(os.listdir(os.path.join(root, dir_name)), '*.edf')
                    if edf_folder in dir_name.upper():
                        path_edf= head_tail[0] + os.sep +"IBIL"+ os.sep + dir_name + os.sep + list_files_edf[0]
                        return True, path_edf

        if not ibil_folder_founded:
            print("IBIL folder not found.")
        return ibil_folder_founded,"none"

    @staticmethod
    def create_combined_pixe(cube_one_pass_pixe,pathlst,num_pass_y,_dict_adc_metadata_arr):
    
        detectors = [134,13,14,34] #"1+3+4","3+4","1+4","1+3"]
        for num_det in detectors:
           
            if num_det == 1234:
                data = cube_one_pass_pixe[0] + cube_one_pass_pixe[1] + cube_one_pass_pixe[2] + cube_one_pass_pixe[3]
                metadata_one_adc = _dict_adc_metadata_arr[0]
            elif num_det == 134:
                data = cube_one_pass_pixe[0] + cube_one_pass_pixe[2] + cube_one_pass_pixe[3]
                metadata_one_adc = _dict_adc_metadata_arr[0]
            elif num_det == 12:
                data = cube_one_pass_pixe[0] + cube_one_pass_pixe[1]
                metadata_one_adc = _dict_adc_metadata_arr[0]
            elif num_det == 34:
                data = cube_one_pass_pixe[2] + cube_one_pass_pixe[3]
                metadata_one_adc = _dict_adc_metadata_arr[2]
            elif num_det == 14:
                data = cube_one_pass_pixe[0] + cube_one_pass_pixe[1] + cube_one_pass_pixe[2]
                metadata_one_adc = _dict_adc_metadata_arr[0]
            elif num_det == 123:
                data = cube_one_pass_pixe[0] + cube_one_pass_pixe[1] + cube_one_pass_pixe[2]
                metadata_one_adc = _dict_adc_metadata_arr[0]
            
            detector_name = ret_adc_name(num_det)
            mysum1 = np.sum(data, axis=0)    # Somme des lignes
            mysum1 = np.sum(mysum1, axis=0)  # Somme des lignes
            mysum1 = np.sum(mysum1, axis=0)  # Somme des lignes
            if mysum1 > 0:
                AGLAEfunction.feed_hdf5_map(data, pathlst, detector_name, num_pass_y,metadata_one_adc)
            

    @staticmethod
    def write_hdf5_metadata(Path_lst, dict_glob_metadata):
        # f = h5py.File('./Data/ReadLst_GZIP.hdf5', 'w')
        head_tail = os.path.split(Path_lst)# Split le Path et le fichier
        destfile = head_tail[1].split(".")
        newdestfile = destfile[0] + ".hdf5"
        index_iba= destfile[0].find("_IBA_")
        index_l1 = destfile[0].find("_L1_")
        index_xrf = destfile[0].find("_XRF1_:")
        det_aglae = ["X0", "X1", "X2", "X3", "X4", "X10", "X11","X12","X13","RBS","RBS150","RBS135","GAMMA","GAMMA70","GAMMA20","IBIL","FORS"]
        iba_para = False
   
        if destfile[1] == 'lst':
            newdestfile = destfile[0] + ".hdf5"


        newdestfile1 =  os.path.join(head_tail[0], newdestfile)

        # print(newdestfile)
        try:
            f = h5py.File(newdestfile1, 'a')

        except:
             f = h5py.File(newdestfile1, 'w')
        try:
            del f["Experimental parameters"]
        except:
            pass
        try:
            del f["stack/detector"]
        except:
            pass
        iba_para = True
        if iba_para == True:
            grp = f.create_group("Experimental parameters")
            root = f['/']
            for exp_attr in dict_glob_metadata:
                 grp.attrs[exp_attr] = dict_glob_metadata[exp_attr]
                 root.attrs[exp_attr] = dict_glob_metadata[exp_attr]

        else:
            grp = f.create_group("parametres")
            grp.attrs["Date"] = "date"
            grp.attrs["Objet"] = "obj"

        f.close()

    @staticmethod
    def write_spectra_hdf5_metadata(Pathfile, dict_glob_metadata):
        # f = h5py.File('./Data/ReadLst_GZIP.hdf5', 'w')
        head_tail = os.path.split(Pathfile)# Split le Path et le fichier
        destfile = head_tail[1].split(".")
        newdestfile = destfile[0] + ".hdf5"
        index_iba= destfile[0].find("_IBA_")
        index_l1 = destfile[0].find("_L1_")
        index_xrf = destfile[0].find("_XRF1_:")
        det_aglae = ["X0", "X1", "X2", "X3", "X4", "X10", "X11","X12","X13","RBS","RBS150","RBS135","GAMMA","GAMMA70","GAMMA20","IBIL","FORS"]
        iba_para = False
   
        if destfile[1] == 'lst':
            newdestfile = destfile[0] + ".hdf5"


        newdestfile1 =  os.path.join(head_tail[0], newdestfile)

        # print(newdestfile)
        try:
            f = h5py.File(newdestfile1, 'a')

        except:
             f = h5py.File(newdestfile1, 'w')
        try:
            del f["Experimental parameters"]
        except:
            pass
        try:
            del f["stack/detector"]
        except:
            pass
        iba_para = True
        if iba_para == True:
            grp = f.create_group("Experimental parameters")
            root = f['/']
            for exp_attr in dict_glob_metadata:
                 grp.attrs[exp_attr] = dict_glob_metadata[exp_attr]
                 root.attrs[exp_attr] = dict_glob_metadata[exp_attr]

        else:
        
            grp = f.create_group("parametres")
            grp.attrs["Date"] = "date"
            grp.attrs["Objet"] = "obj"

        f.close()


    @staticmethod
    def finalhdf5(Pathfile,detname):
        head_tail = os.path.split(Pathfile)  # Split le Path et le fichier
        destfile = head_tail[1].split(".")
        newdestfile = destfile[0] + ".hdf5"
        index_iba = destfile[0].find("_IBA_")
        index_l1 = destfile[0].find("_L1_")
        index_xrf = destfile[0].find("_XRF1_:")
        det_aglae = ["LE0", "HE1", "HE2", "HE3", "HE4", "HE10", "HE11", "HE12", "HE13", "RBS", "RBS150", "RBS135",
                     "GAMMA", "GAMMA70", "GAMMA20", "IBIL", "FORS"]
        iba_para = False

        for det1 in det_aglae:
            if detname == det1:
                iba_para = True

        Myname = []
        if destfile[1] == 'lst':
            newdestfile = destfile[0] + "_IBA.hdf5"
        elif destfile[1] == 'edf':
            # n = len(destfile[0])
            name = os.path.basename(destfile[0])
            if index_iba > 0:
                Myname = name.split('_IBA')
                FinalHDF = Myname[0] + "_IBA.hdf5"
            else:
                Myname = name.split('_')
                FinalHDF = Myname[0] + "_" + Myname[1] + "_IBA.hdf5"

        return FinalHDF

    @staticmethod
    def open_header_lst(pathlst:str):
        import os
        # pathlst = "E:/21mai0106.lst"
        tmpheader = ""
        para2 = ""
        header1 = list()
                
        indexadc = -1
        adc_readed = np.array([0,0,0,0,0,0,0,0,0,0,0,0],dtype=bool)
        head_tail = os.path.split(pathlst)  # Split le Path et le fichier
        root_text = os.path.splitext(head_tail[1])  # Split le fichier et ext
        dict_adc_metadata_arr = np.full((20), {})
        dict_metadata_global = {}
        dict_metadata= {}
        toto = []
        end_adc = False
                
        datainname = root_text[0].split("_")
        if len(datainname) > 4:
            dateacq = datainname[0]
            num_analyse = datainname[1]
            objetacq = datainname[2]
            projetacq = datainname[3]
        else:
            dateacq = "?"
            objetacq = "?"
            projetacq = "?"

        # dict_metadata_global["obj"] = objetacq
        # dict_metadata_global["num analyse"] = num_analyse
        # dict_metadata_global["prj"] = projetacq
        # dict_metadata_global["dateacq"] = dateacq #En général la date acquisition -> 20250105''
        # dict_metadata_global["type analyse"] = "OBJ"
        t =0
        det_aglae = ["X0", "X1", "X3", "X4", "X10", "X11", "X12", "X13", "RBS", "RBS150", "RBS135",
                     "GAMMA", "GAMMA70", "GAMMA20", "IBIL", "FORS"]
        
        with open(pathlst, "r",errors='ignore') as file_lst:
            import os
            size_lst = os.path.getsize(pathlst)
            fin_header =False
            while fin_header ==False: 
                tmpheader = file_lst.readline()
                t+=1
                if t == 495:
                     t=t
                else:
                    t=t           
                # Map size:1280,1280,64,64,0,2564,100000_
                if "[ORS" in str.upper(tmpheader) or "[MAP" in str.upper(tmpheader): # Fin lecture ADC
                    end_adc = True


                if "[ADC" in str.upper(tmpheader): # Read metadata ADC1
                   mynumero = re.search(r'\[ADC(\d+)\]', tmpheader) #re.search(r'[ADC(\d+)]', tmpheader) 
                   indexadc = int(mynumero.group(1)) -1
                   dict_metadata.clear() #dict_metadata_arr[indexadc]
                   idx_metadata = 0
                   adc_readed[indexadc] = 1
            
                if indexadc == 0 and "REPORT-FILE from  written" in str(tmpheader): 
                    para = str.split(str(tmpheader), sep='written')
                    dict_metadata_global["timestamp"]= AGLAEfunction.clean_text(para[1])

                if "cmline1=" in str(tmpheader): # Nom detecteur
                    para = str.split(str(tmpheader), sep='=')
                    text_adc_name = AGLAEfunction.clean_text(para[1])
                    text_adc_name= text_adc_name.upper()
                    if text_adc_name in det_aglae:
                        dict_metadata["adc name"]= text_adc_name
                    else:
                        dict_metadata["adc name"]= "det. OFF"
                    

                    
#cmline2= detector:SDD Ketek ,dia:50mm2 ,window:4um Be, S/N:xxxxx, Angle:50° 
#cmline2= detector:pips ,dia:50mm2 ,window:4um Be, S/N:xxxxx, Angle:50° 

                if "DET:" in str.upper(tmpheader): # Info detecteur
                    para = str.split(str(tmpheader), sep='=')
                    para2 = str.split(para[1], sep=',') # 
                    par_det= str.split(para2[0], sep=':')
                    for i, text in enumerate(para2):
                        text = AGLAEfunction.clean_text(text)
                        text2 = str.split(text, sep=':')[1]
                        if "DET:" in str.upper(text):
                            dict_metadata["det. type"] = str.split(text, sep=':')[1]
                        if "AREA:" in str.upper(text):
                            dict_metadata["det. active area"]= str.split(text, sep=':')[1]
                        if "TCK:" in str.upper(text):
                            dict_metadata["det. thickness"] =str.split(text, sep=':')[1]
                        if "WIN:" in str.upper(text):
                            dict_metadata["det. entrance window"] =str.split(text, sep=':')[1]
                        if "ANGLE:"in str.upper(text) :
                            dict_metadata["det. angle"] =str.split(text, sep=':')[1]
                        if  "S/N:" in str.upper(text): 
                            dict_metadata["det. S/N"] =str.split(text, sep=':')[1]
                
                if "FILTER:" in str.upper(tmpheader): 
                    para = str.split(str(tmpheader), sep='=')
                    para2 = str.split(para[1], sep=',')
                    #para = str.split(str(tmpheader), sep=':')
                    #text = AGLAEFile.clean_text(para[1])  #remplace µ par u et supprime \\r\\n
                    for i, text in enumerate(para2):
                        text = AGLAEfunction.clean_text(text)
                        #dict_metadata["filter"] =text
                        if "WIN:" in str.upper(text):
                            dict_metadata["det. entrance window"] =str.split(text, sep=':')[1]
                        if "ANGLE:"in str.upper(text) :
                            dict_metadata["det. angle"] =str.split(text, sep=':')[1]
                        if "FILTER:"in str.upper(text) :
                            dict_metadata["det. filter"] =str.split(text, sep=':')[1]
                        if  "S/N:" in str.upper(text): 
                            dict_metadata["det. S/N"] =str.split(text, sep=':')[1]

                if "CALIBRATION:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])
                    dict_metadata["calibration"] = text

                if "INSTITUTION:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])  #remplace µ par u et supprime \\r\\n
                    dict_metadata_global["institution"] = text

                if "SAMP. INFO:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])
                    dict_metadata_global["analyse description"] = text

                if "REF ANALYSE:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])
                   # if "STANDARD:" in str.upper(text): #if text == "satndar":
                    dict_metadata_global["ref. analyse"] = text
                   # else:
                    #    dict_metadata_global["ref. object"] = text
                        

                if "USERNAME:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])  
                    dict_metadata_global["username"] =text

                if "PRJ EUPHROSYNE:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])  
                    dict_metadata_global["prj euphrosyne"] =text

                if "OBJ EUPHROSYNE:" in str.upper(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    text = AGLAEfunction.clean_text(para[1])  
                    dict_metadata_global["obj euphrosyne"] =text
                    if str.upper (text) == "STANDARD":
                        dict_metadata_global["target type"] = "standard"
                    else:
                        dict_metadata_global["target type"] = "object"


                if "MAP SIZE:" in str.upper(tmpheader):
                    para2 = str.split(str(tmpheader), sep=':')
                    para2 = str.split(para2[1], sep=",")
                    map_info = ["map size x (um)","map size y (um)","pixel size x (um)","pixel size y (um)","pen size (um)","dose/column","dose" ]
                    for i, text in enumerate(para2): 
                        text = AGLAEfunction.clean_text(text)
                        para2[i] = text
                        dict_metadata_global[map_info[i]] = text
                        
    #cmline7= calibration: MCA a= 1, MCA b= 1, MCA c= 0
              
                #cmline9= Exp.Info:Proton , 2976 keV        
                if "EXP.INFO" in str.upper(tmpheader): 
                    para2 = str.split(str(tmpheader), sep=':')
                    para2 = str.split(para2[1], sep=",")
                    beam_info = ["particle", "beam energy"]

                    for i, text in enumerate(para2): 
                        text = AGLAEfunction.clean_text(text)
                        dict_metadata_global[beam_info[i]] = text
                                     
             
                
                if np.any(adc_readed) == True and end_adc == False:
                    dict_adc_metadata_arr[indexadc] = dict_metadata.copy()
                    
                
                #toto.append(dict_metadata.copy())
                
                fin_header = '[LISTDATA]' in str.upper(tmpheader) or t > 5000
                # header1.append(tmpheader)
        # Info IBIL en constant
        dict_metadata= {}
        dict_metadata["spectro name"] = "IBIL"
        dict_metadata["calibration"] = "MCA a= 0.801, MCA b= 189.0, MCA c= 0"
        dict_metadata["spectro. S/N"]= "unknown"
        dict_metadata["spectro. angle"] = "45"
        dict_metadata["spectro. slit size"] = "100 um"
        dict_metadata["fiber diameter"] = "1000 um"
        dict_metadata["fiber S/N"] = "unknown"
        dict_metadata["spectro. type"] = "QE65000 Ocean Optics"
        dict_adc_metadata_arr[12] = dict_metadata.copy() # GAMMA
        return dict_adc_metadata_arr,dict_metadata_global
    
    def clean_text(text):
        text = text.replace("\\xb5", 'u')
        text = text.replace("\r\n", '')
        text = text.replace('\n', '')
        text = text.replace("\\", 'jj')
        text = text.strip()
                        
        return text
       
    def return_adc_adjusted_index(data_array_previous,data_array):

        data_array = data_array[data_array != 65535]
        data_array = np.append(data_array_previous, data_array)
        min_last_pos_x_y_in_array = np.shape(data_array)
        shape_data_array = int(min_last_pos_x_y_in_array[0])
        # Recherche de la valeur 0x8000 (32768) dans le tableau de donnees
        indices_32768 = np.where(data_array == 32768)
        indices_32768 = np.array(indices_32768[0])
        indices_32768 = np.delete(indices_32768, len(indices_32768) - 1)
        # Cr�ation d'indices ajust�s et filtrage
        one_array = np.full(np.shape(indices_32768), 1)
        adjusted_indices = indices_32768 - one_array
        return adjusted_indices, data_array ,shape_data_array

    def return_index_adc_in_data_array(adjusted_indices,adc_values,num_line_adc,conditionXY):
        """ Retourne tout les indices des ADC dans DataArray"""
                # Op�rations bitwise et filtrage des indices
        adc_masked = np.bitwise_and(adc_values[:], 0b0000000000000001 << num_line_adc)
        conditionADC = (adc_masked != 0) # and coord_X_masket[:] != 0 and coord_Y_masket[:] != 0
        condition_final = np.logical_and(conditionADC, conditionXY)
        filtered_indices = np.where(condition_final, adjusted_indices[:],
                                     np.full(len(condition_final), -1, dtype=np.int16))
      
        non_zero_indices = filtered_indices[filtered_indices != -1]
        if len(non_zero_indices) < 10:
           return [-1]
        # Convert `non_zero_indices` to a NumPy array for better performance
        non_zero_indices = np.array(non_zero_indices)
        return non_zero_indices



    def return_val_to_read(adc_words,non_zero_indices):
        ind10 = []
        bit_count_array = np.empty(len(adc_words), dtype=np.uint32)
        qui_a_declanche = np.empty((12, len(adc_words)), dtype=np.uint32)
        # Loop through each bit position and count the bits set to 1
        for bit_position in range(12):
            # Create a mask for the current bit position
            bit_mask = 0b0000000000000001 << bit_position
            # Use bitwise AND to check if the current bit is set
            adc_declanche = adc_words & bit_mask
            bit_count_array += adc_declanche > 0
            ind10 = np.where(adc_declanche != 0, non_zero_indices + 1, np.zeros(len(adc_declanche)))
            qui_a_declanche[bit_position, :] = ind10

        del ind10

        compteur_valeur = np.empty((12, len(adc_words)), dtype=np.uint8)
        mysum = np.zeros((1, len(adc_words)), dtype=np.uint8)
        # for arr in qui_a_declanche:
        #          zero_els = jnp.count_nonzero(arr == 0)
        del adc_words

        for x in range(12):  # 15, -1,-1):
            # r= np.where(qui_a_declanche[x,:] !=0, qui_a_declanche[x,:] - bit_count_array[:],qui_a_declanche[x,:])
            compteur_valeur[x, :] = np.where(qui_a_declanche[x, :] != 0, mysum + 1, qui_a_declanche[x, :])
            mysum = np.where(compteur_valeur[x, :] != 0, mysum + 1, mysum)

        indice_val_to_read = qui_a_declanche + compteur_valeur  # np.full(len(compteur_valeur),1)
        return indice_val_to_read

    def read_min_max_y(coord_y):
        """ recherche la derniere valeur de Y du tableau en Input"""
        last_pos_y = np.empty(0, dtype=np.uint16)
        if len(coord_y) > 300:
            r1 = 300
        else:
            r1 = len(coord_y) -1

        for pos in range(r1):
            last_pos_y = np.append(last_pos_y, coord_y[-pos+1])

        max_val_y_lue = np.max(last_pos_y)
        min_val_y_lue = np.min(last_pos_y)
        nb_max_value = np.count_nonzero(last_pos_y == max_val_y_lue)
        if nb_max_value > 2:
            max_val_y_lue = max_val_y_lue
        else:
            max_val_y_lue = min_val_y_lue


        return max_val_y_lue,min_val_y_lue

    def range_x(coord_x,croissant):
        
        local_max_x = np.max(coord_x)
        local_min_x = np.min(coord_x)
        nb_max = np.where(coord_x == local_max_x)
        nb_min = np.where(coord_x == local_max_x)
        if len(nb_max[0] > 10):
            last_x_value = local_max_x
        
        if len(nb_min[0] > 10):
            first_x_value = local_min_x

        
        return first_x_value, last_x_value

   
    def read_range_x(coord_x,croissant):
        local_max_x = np.max(coord_x)
        last_pos_x = np.empty(0, dtype=np.uint16)
        first_pos_x = np.empty(0, dtype=np.uint16)
        if len(coord_x) > 100:
            r1 = 100
        else:
            r1 = len(coord_x) -1

        for pos in range(r1):
            if croissant == True:
                last_pos_x = np.append(last_pos_x, coord_x[-pos-1])  # A partir de la fin
                first_pos_x = np.append(first_pos_x, coord_x[pos])  # A partir du d�but
            else:
                last_pos_x = np.append(last_pos_x, coord_x[pos])  # A partir du d�but
                first_pos_x = np.append(first_pos_x, coord_x[-pos-1])  # A partir de la fin

        last_pos_x = np.delete(last_pos_x, 0)
        count_x = np.bincount(last_pos_x)
        last_x_value = int(np.shape(count_x)[0]) - 1 # On enleve la derni�re colonne

        
        count_x_min = np.bincount(first_pos_x)
        first_x = np.where(count_x_min == max(count_x_min))
        s1 = np.shape(first_x)
        if len(s1) ==1: 
            first_x_value = int(first_x[0])
        if len(s1) ==2: 
            first_x_value = int(first_x[0][0]) # plusieurs valeur superieur X identique 

        return first_x_value, last_x_value


    def get_X_Y_condition(adc_values,ADC_X,ADC_Y):
            coord_X_masket = np.bitwise_and(adc_values[:], 0b0000000000000001 << ADC_X)
            coord_Y_masket = np.bitwise_and(adc_values[:], 0b0000000000000001 << ADC_Y)
            conditionX = coord_X_masket[:] != 0
            conditionY = coord_Y_masket[:] != 0
            conditionXY = np.logical_and(conditionY, conditionX)
            return conditionXY      

    def read_indice_max_x(croissant,sizeX,coord_x,included_x):
           
            indice_x_max = np.where(coord_x == included_x)
            len_x_max = len(np.shape(indice_x_max[0]))
            if len_x_max < 1: #Pas ce X dans le Dataarray , faisceau OFF ?"

                while len_x_max < 1:
                    if croissant == True:
                        included_x -=1
                    else:
                        included_x +=1
                    indice_x_max = np.where(coord_x == included_x)
                    len_x_max = len(np.shape(indice_x_max[0]))    

            if croissant == True:
                try:
                    indice_x_last = indice_x_max[0][-1]
                except:
                    indice_x_last = indice_x_max[-1]
            else:
                try:
                    indice_x_last = indice_x_max[0][-1]
                except:
                    indice_x_last = indice_x_max[-1]
           
            return indice_x_last
        
    
    def read_min_x(coord_x, croissant, previous_last_x):

        last_pos_x = np.empty(0, dtype=np.uint16)
        first_pos_x = np.empty(0, dtype=np.uint16)
        for pos in range(100):

            if croissant == True:
                last_pos_x = np.append(last_pos_x, coord_x[-pos-1])  # A partir de la fin
                first_pos_x = np.append(first_pos_x, coord_x[pos])  # A partir du d�but
            else:
                last_pos_x = np.append(last_pos_x, coord_x[pos])  # A partir du d�but
                first_pos_x = np.append(first_pos_x, coord_x[-pos-1])  # A partir de la fin

        last_pos_x = np.delete(last_pos_x, 0)
        count_x = np.bincount(last_pos_x)
        last_x = int(np.shape(count_x)[0]) - 1  # On enleve la derni�re colonne
        if len(first_pos_x) == 0:
            count_x_min = 0
        else:
            try:
                count_x_min = np.bincount(first_pos_x)
                first_x = np.where(count_x_min == max(count_x_min))
                first_x_value = int(first_x[0])

            except:
                first_x_value = 0

        return first_x_value

    def read_max_indice_change_colonne(coord_y,y_scan_total):
        fin_ligne = False
        indice_y_last = np.where(coord_y > y_scan_total)  # recherche les val de la dernier colonne
        if len(indice_y_last[0]) < 50:
            fin_ligne = True
            indice_last = 0
        indice_last = indice_y_last[0][0] - 1
        return indice_last


    def get_colums_range(croissant,first_x_value,last_x_value,included_x,end_lst_file_found):
        """True si plusieurs column dans le Data array"""
        if croissant== True:
            if end_lst_file_found == False :
                columns = included_x > first_x_value
            else:
                columns = included_x > first_x_value
        
        else:
            if first_x_value < last_x_value :
                if end_lst_file_found == False :
                    columns = included_x < last_x_value
                else:
                    columns = included_x < last_x_value
            else:
                columns = False
        return columns



    def get_last_x_to_include(croissant,columns,last_x_value,first_x_value,change_line,fin_lst):
        """Recupére Last X à inclure dans ce process"""
        # if columns== False:
        #     included_x = last_x_value
        if columns==True and change_line==False and fin_lst==False:
            if croissant== True:
                included_x = last_x_value -1 #on va eclure le X-1
            elif croissant== False:
                included_x = first_x_value +1
        else:
            if croissant== True:
                included_x = last_x_value
            else:
                included_x = first_x_value
            
        return included_x
    

    def clean_coord(sizeX,sizeY,coord_x,coord_y,b_previous_find_x,previous_find_x,croissant):
        error_y =False
        max_size_x = ret_range_bytes(sizeX - 1)
        max_size_y = ret_range_bytes(sizeY - 1)
        coord_x = coord_x & max_size_x  # & binaire pour masquer bits > max_size à 0
        c1 = coord_y
        c1 = c1[c1 != 0]
        
        if len(c1) < 100:
            error_y = True
        coord_y = coord_y & max_size_y 

        # Met des -1 aux coord X et Y > valeur de la carto
        val_out_range_x = np.where(coord_x > sizeX - 1)
        coord_x = np.delete(coord_x, val_out_range_x)
        coord_y = np.delete(coord_y, val_out_range_x)
        if b_previous_find_x == True:
            val_out_range_x = np.where(coord_x == previous_find_x)
            coord_x = np.delete(coord_x, val_out_range_x)
            coord_y = np.delete(coord_y, val_out_range_x)
  
        return coord_x,coord_y,error_y
    



    def extract_lst_vector(path_lst:str, dict_para_global:dict,_dict_adc_metadata_arr:dict):
        pathlst1 = path_lst
        _dict_channel_adc,_dict_config_mpawin_adc,_dict_combined_adc = read_cfg_adc(resource_path("config_lst2hdf5.ini"))
        print(_dict_channel_adc)
       
        
        #array_adc = [0,2,3,4,6,7,10,11]
        array_adc = []
        for key,value in _dict_config_mpawin_adc.items():
               if str.upper(value) == "COORD_X":
                   ADC_X= int(key)
               elif str.upper(value) == "COORD_Y": 
                   ADC_Y= int(key)
               elif str.upper(value) != "OFF":
                   array_adc.append(int(key))
               
        tmpheader = ""
        header1 = list()
        sizeX = 1
        sizeY = 1
        #print(path_lst)
        print("map size = ",dict_para_global['map size x (um)'], dict_para_global['map size y (um)'])
        
        sizeX = int(dict_para_global['map size x (um)']) / int(dict_para_global['pixel size x (um)'])
        sizeY = int(dict_para_global['map size y (um)']) / int(dict_para_global['pixel size y (um)'])
        sizeX = int(sizeX)
        sizeY = int(sizeY)
        adcnum = []
        

        nbcanaux = 1024
        nbcanaux_pixe = int(_dict_channel_adc['pixe'])
        nbcanaux_gamma20 = int(_dict_channel_adc['gamma20'])
        nbcanaux_gamma70 = int(_dict_channel_adc['gamma70'])
        nbcanaux_rbs = int(_dict_channel_adc['rbs'])
        cube = np.zeros((sizeX, sizeY, nbcanaux), 'u4')
        # for i in range (0,50):

        file = open(path_lst, 'rb')
        size_lst = getSize(file)
        file.close()
        allheader = ""
        fin_header = False
        t=0
        error_in_lst_file = False
        with open(path_lst, "rb") as file_lst:  # Trop long
            while fin_header == False: #tmpheader != b'[LISTDATA]\r\n':
                tmpheader = file_lst.readline()
                tmp1 = str(tmpheader)
                allheader = allheader + tmp1.replace("\\r\\n", '')
                t+=1    
                size_lst -= len(tmp1) - 2
                # if "condition" in tmpheader.decode('utf-8', errors='ignore'):
                #     toto = 1
              
                fin_header = tmpheader == b'[LISTDATA]\r\n' or tmpheader == b'[LISTDATA]\n' or t > 5000
                # header1.append(tmpheader)
   
            pensize = int(dict_para_global["pen size (um)"])
            nb_pass_y = int(sizeY / (pensize / int(dict_para_global["pixel size y (um)"])))
            nb_column_total = sizeX*nb_pass_y
           
        
   
            size_lst = int(size_lst)  # car on lit des Uint16 donc 2 fois moins que le nombre de bytes (Uint8)
            size_block = size_lst
            size_one_scan = size_lst / (nb_pass_y+1)
            size_4_column_scan = (size_one_scan / (sizeX)) * 2 #/(sizeX/40))  # taille 4 column
            size_block = int(size_4_column_scan)
            size_block_big = int(size_4_column_scan) * 2
            large_map = False

            if size_lst > 1000000 and sizeX > 40:
                large_map = True
       
            
            if nb_pass_y == 1 and size_lst < 50*10**6: 
                size_block = size_lst
            # size_block = 100000
            nb_read_total = 0
            
            
            if size_lst > size_block:
                nb_loop = int(size_lst / size_block)
                nb_loop += 1
                reste = size_lst % size_block
            else:
                nb_loop = 1
                reste = 0
                nb_pass_y = 1

            nb_byte_to_read = size_block
            indice_in_datablock = 0
            nb_read_total = 0
            max_val_y_lue = 0
            croissant = True
            y_scan = int(sizeY / nb_pass_y) - 1
            nb_column = int(sizeY / nb_pass_y)
            data_array_previous = np.empty(0, dtype=np.uint16)
            end_lst_file_found = False
            last_x_maps = 0
            columns = True
            nb_adc_not_found = 0
            
            if nb_pass_y % 2 == 0:
                last_x_maps = 0
            else:
                last_x_maps = sizeX - 1
            nb_total_event = 0
            nb_event_x0 = 0 
            for num_pass_y in range(nb_pass_y):
                print(num_pass_y ,"//",nb_pass_y,end='\n')

                if end_lst_file_found==True: # fin LST avant fin de la taille de la carto (ABORT sur New Orion)
                    break   
                if (num_pass_y % 2 == 0):
                    croissant = True
                    next_x_value = np.zeros(12, dtype=np.uint16)

                else:
                    croissant = False
                    next_x_value = np.full(12, sizeX - 1, dtype=np.uint16)


                end_lst_file_found = False
                y_max_current_scan = y_scan + (num_pass_y * nb_column)

                cube_one_pass_pixe = np.empty((5, nb_column, sizeX, nbcanaux_pixe), dtype=np.uint32)
                cube_one_pass_gamma20 = np.empty((1, nb_column, sizeX, nbcanaux_gamma20), dtype=np.uint32)
                cube_one_pass_gamma70 = np.empty((1, nb_column, sizeX, nbcanaux_gamma70), dtype=np.uint32)
                cube_one_pass_rbs = np.empty((3, nb_column, sizeX, nbcanaux_rbs), dtype=np.uint32)

                fin_ligne = False
                change_line= False
                zero_off = False
                previous_find_x = 0
                b_previous_find_x = False
                    
                while (fin_ligne == False and end_lst_file_found == False):  # max_val_y_lue <= y_scan): # or croissa nte == False ):

                    adc2read = 0
                    adc_values = np.empty(0, dtype=np.uint16)
                    data_array = np.empty(0, dtype=np.uint16)
                    adjusted_indices = np.empty(0, dtype=np.uint16)
                    adjusted_indices_previous = np.empty(0, dtype=np.uint16)
                    # if large_map == True:
                    #     if croissant == True:
                    #         if previous_find_x < sizeX -20 :
                    #             nb_byte_to_read = size_block_big
                    #         else:
                    #             nb_byte_to_read = size_block

                    #     if croissant == False: 
                    #         if previous_find_x >20 or previous_find_x==0:
                    #             nb_byte_to_read = size_block_big
                    #         else:
                    nb_byte_to_read = size_block
                              
                    min_last_pos_x_y_in_array = 0 #nb_byte_to_read
                    data_array = np.fromfile(file_lst, dtype=np.uint16, count=int(nb_byte_to_read))
        
                    if len(data_array) < nb_byte_to_read:
                        end_lst_file_found = True
                        print("\n End LST file found",end='\n' )
                    
                    nb_32768 = np.count_nonzero(data_array == 32768)
                    nb_total_event = nb_total_event + nb_32768
                    adjusted_indices, data_array ,shape_data_array = AGLAEfunction.return_adc_adjusted_index (data_array_previous, data_array)
                    adc_values = np.array(data_array[adjusted_indices])
                    if len(data_array) < 1 : 
                        exit 

                    nb_read_total += (nb_byte_to_read * 2) + len(data_array_previous)
                    t1 = perf_counter()
                    
                    #array_adc = [0,4]
                    max_size_x = ret_range_bytes(sizeX - 1)
                    max_size_y = ret_range_bytes(sizeY - 1)
                    conditionXY= AGLAEfunction.get_X_Y_condition(adc_values,ADC_X,ADC_Y)
                    nb_adc_not_found = 0
                    last_indx_x = 0
                   
                    for num_line_adc in array_adc: #range(12):
                        sleep(0.002)
                        if num_line_adc == 1 or num_line_adc == 8 or num_line_adc == 9 or num_line_adc == 55: continue

                        switcher = {5: nbcanaux_pixe, 0: nbcanaux_pixe, 1: nbcanaux_pixe, 2: nbcanaux_pixe, 3: nbcanaux_pixe, 4: nbcanaux_pixe, 80: nbcanaux_pixe, 81: nbcanaux_pixe,
                                    82: nbcanaux_pixe, 6: nbcanaux_rbs, 7: nbcanaux_rbs, 10: nbcanaux_gamma20, 11: nbcanaux_gamma70}

                        nbcanaux = switcher.get(num_line_adc)
                        detector = ret_adc_name(num_line_adc)
                        adc2read = num_line_adc + 1
                        # adc2read = ret_num_adc(self.detector)
                        t0 = perf_counter()
                        # Return 
                        non_zero_indices = AGLAEfunction.return_index_adc_in_data_array(adjusted_indices,adc_values,num_line_adc,conditionXY)
                        if non_zero_indices[0] == -1 or len(non_zero_indices) < 50:
                            nb_adc_not_found +=1
                            continue
                        adc_words = data_array[non_zero_indices]
                        indice_val_to_read = AGLAEfunction.return_val_to_read(adc_words,non_zero_indices)

                        coord_x = data_array[indice_val_to_read[ADC_X, :]]  
                        #coord_x = coord_x & max_size_x  # & binaire pour masquer bits > max_size à 0
                        coord_y = data_array[indice_val_to_read[ADC_Y, :]] 
                     
                        # if croissant == False:
                        #      coord_x = np.flip(coord_x)
                        #      coord_y = np.flip(coord_y)
                             
                        #if end_lst_file_found == False :
                        coord_x, coord_y ,error= AGLAEfunction.clean_coord(sizeX,sizeY,coord_x,coord_y,b_previous_find_x,previous_find_x,croissant) # del 
                        max_val_y_lue,min_val_y_lue = AGLAEfunction.read_min_max_y(coord_y)
                        change_line = look_if_next_line(max_val_y_lue,y_max_current_scan) #True or False si Changement de Y sup.
                        
                        if coord_x[0] !=0 and change_line == False: # coord_X =0 anormal si changement de ligne je garde les zeros (Y max sert de delimiateteur et non X)
                            x_zero_error = np.where(coord_x ==0) # Recherche si des X=0 présents dans X !=0
                        else:
                            x_zero_error = np.zeros((1, 1), dtype=np.uint32)

                        max_val_y_lue,min_val_y_lue = AGLAEfunction.read_min_max_y(coord_y)
                        first_x_value, last_x_value = AGLAEfunction.read_range_x(coord_x, croissant)
                        #first_x_value, last_x_value = AGLAEFile.range_x(coord_x, croissant)
                      
                        if first_x_value !=0 and len(x_zero_error[0]) > 1: #coord_X =0 anormal
                            coord_x = np.delete(coord_x, x_zero_error)
                            coord_y = np.delete(coord_y, x_zero_error)
                            first_x_value, last_x_value = AGLAEfunction.read_range_x(coord_x, croissant)
                                
                        change_line = look_if_next_line(max_val_y_lue,y_max_current_scan) #True or False si Changement de Y sup.
                        val_x_fin_map = get_x_end_line_scan(croissant,sizeX) # retourne val final 0 ou SizeX-1
                        
                        if change_line == False:
                            fin_lst = look_if_end_lst(max_val_y_lue,sizeY,val_x_fin_map,last_x_value)
                        elif change_line == True:
                            if croissant:
                                last_x_value = sizeX - 1
                            else:
                                first_x_value = 0
                        else:
                            fin_lst = False # change line -> fin lst impossible

                       
                       # Dertermine la dernière valeur X
                        included_x = AGLAEfunction.get_last_x_to_include(croissant, columns, last_x_value,first_x_value,change_line,fin_lst)
                        columns= AGLAEfunction.get_colums_range(croissant,first_x_value,last_x_value,included_x,end_lst_file_found)
                        included_x = AGLAEfunction.get_last_x_to_include(croissant, columns, last_x_value,first_x_value,change_line,fin_lst)
                                                   
                        if last_x_value < first_x_value and croissant==True: # Cas trop lus de columns
                            last_x_value = sizeX-1
                            columns= AGLAEfunction.get_colums_range(croissant,first_x_value,last_x_value,included_x,end_lst_file_found)
                        
                        if first_x_value > last_x_value and croissant==False: # Cas trop lus de columns
                            first_x_value = 0
                            columns= AGLAEfunction.get_colums_range(croissant,first_x_value,last_x_value,included_x,end_lst_file_found)
           
                       
                        if end_lst_file_found == True or fin_lst == True:
                                 indice_last = len(coord_y) -1
                        else:
                            if columns == True and change_line == False: # plus de 1 colonne
                                indice_last = AGLAEfunction.read_indice_max_x(croissant,sizeX,coord_x,included_x)#,next_x_value[num_line_adc])
                            elif columns == False and change_line == False: # 1 Colonne
                                indice_last = AGLAEfunction.read_indice_max_x(croissant,sizeX,coord_x,included_x)#,next_x_value[num_line_adc])
                            elif change_line == True:
                                indice_last = AGLAEfunction.read_max_indice_change_colonne(coord_y,y_max_current_scan) #Recherche last_indice avec Y < scan total
                                if croissant == True:
                                    included_x = sizeX-1
                                else:
                                    included_x = 0     
                                fin_ligne = True
                           
                       
                        if num_line_adc== 0 :
                          if croissant == True:
                              print("X:", first_x_value,"To",included_x,end=",")
                              sleep(0.02)
                              
                              
                          else:
                              print("X:", last_x_value,"To",included_x,end=",")
                              sleep(0.02)
                              if included_x == 25:
                                  included_x = 25
                        
                        max_data_array = indice_val_to_read[ADC_X, indice_last]
                        coord_x = coord_x[:indice_last]
                        coord_y = coord_y[:indice_last]


                        if max_data_array > min_last_pos_x_y_in_array:
                                min_last_pos_x_y_in_array = max_data_array

                 
                        non_zero_indices = np.nonzero(indice_val_to_read[num_line_adc, :indice_last])
                        if len(non_zero_indices[0]) < 2:  # pas de valeur pour cet adc dans ce Block de Data Array
                            continue

                        adc1 = data_array[indice_val_to_read[num_line_adc, non_zero_indices]]
                        adc1 = np.array(adc1 & nbcanaux - 1)
                        if num_pass_y != 0:
                            coord_y = coord_y - (num_pass_y * nb_column)

                        new_coord_x = coord_x [non_zero_indices]
                        new_coord_y = coord_y [non_zero_indices]

                        #if (croissant == True and last_x_value==sizeX-1) or (croissant == False and last_x_value == 0):
                        if (croissant == True and included_x==sizeX-1) or (croissant == False and included_x == 0):
                            fin_ligne = True
                                                                      
                        if croissant == True and end_lst_file_found == True: # Si carto stopper avant fin de la ligne
                            p2 = last_x_value # Je prend la dernier column en compte dans mon histogramme
                            p1 = first_x_value
                        if croissant == False and end_lst_file_found == True: # Si carto stopper avant fin de la ligne
                            p2 = first_x_value # Je prend la dernier column en compte dans mon histogramme
                            p1 = last_x_value
                        elif croissant == False:
                            p2 = last_x_value
                            p1 = included_x
                        elif croissant == True:
                            p2 = included_x #last_x_value -1
                            p1 = first_x_value

                       
                        if croissant == True:
                            adc3 =adc1[0]
                            del adc1
                         
                            if columns == False:
                                range_histo = 1
                            else:
                                r1 = [p1, p2]
                                range_histo = (p2 - p1) + 1


                        else:
                            new_coord_x = np.delete(new_coord_x, 0)
                            new_coord_y = np.delete(new_coord_y, 0)

                            adc2 = np.delete(adc1[0], 0)
                            adc3 = np.flip(adc2)
                            del adc1

                            if columns == False:
                                range_histo = 1
                            elif p2>p1:
                                r1 = [p1, p2]
                                range_histo = (p2 - p1) + 1
                            else:
                                range_histo = 1
                        if num_line_adc == 4:
                            nb_event_x0 = nb_event_x0 + len(adc3)

                        if range_histo < 0:
                            print('error range_histo') 
                        if range_histo==1:
                           H1, xedges, yedges= np.histogram2d(new_coord_y,adc3,bins=(nb_column,nbcanaux),range= ({0, nb_column-1},{0, nbcanaux-1}))
                        
                        else:
                            # H1, edges = np.histogramdd((new_coord_y, new_coord_x, adc3),
                            #                        range=({0, nb_column-1}, r1, {0, nbcanaux-1}),
                            #                        bins=(nb_column, range_histo, nbcanaux))
                            H1, edges = np.histogramdd((new_coord_y, new_coord_x, adc3),
                                                   range=({0, nb_column}, r1, {0, nbcanaux}), 
                                                   bins=(nb_column, range_histo, nbcanaux))

        
                                    
                        if croissant == True:
                            ind_1 = first_x_value
                            ind_2 = included_x + 1 # Numpy array exclu le dernier indice
                        else:
                            ind_1 = included_x 
                            ind_2 =  last_x_value + 1 # Numpy array exclu le dernier indice


                        if first_x_value == 0:
                            first_x_value=0
                        if num_line_adc <=4: # PIXE HE
                            if range_histo == 1:
                                cube_one_pass_pixe[num_line_adc][:, ind_1, :] = H1
                            else:
                                cube_one_pass_pixe[num_line_adc][0:,ind_1:ind_2, 0:] = H1

                        elif num_line_adc == 5 or num_line_adc == 6 or  num_line_adc == 7: # RBS ou RBS135 ou RBS150    
                            if range_histo == 1:
                                cube_one_pass_rbs[num_line_adc - 5][:, ind_1, :] = H1
                            else:
                                cube_one_pass_rbs[num_line_adc - 5][0:,ind_1:ind_2, 0:] = H1


                        elif num_line_adc == 10: # Gamma20
                            if range_histo == 1:

                                cube_one_pass_gamma20[0][0:, int(next_x_value[num_line_adc]),0:] = H1
                            else:
                                cube_one_pass_gamma20[0][0:,first_x_value:last_x_value, 0:] = H1
                                cube_one_pass_gamma20[0][0:,ind_1:ind_2, 0:] = H1

                        elif num_line_adc == 11: # Gamma70
                            if range_histo == 1:
                                cube_one_pass_gamma70[0][0:, first_x_value,0:] = H1
                            else:
                               # cube_one_pass_gamma70[0][0:,first_x_value:last_x_value, 0:] = H1
                                cube_one_pass_gamma70[0][0:,ind_1:ind_2, 0:] = H1
                                               
                      
                        if range_histo != 1 and croissant == True:
                            next_x_value[num_line_adc] = last_x_value
                        else:
                            next_x_value[num_line_adc] = first_x_value
                        
                    if nb_adc_not_found < 9:
                        if first_x_value ==0 and croissant == True:
                            zero_off = True     
 
                        if min_last_pos_x_y_in_array < int(shape_data_array):
                            data_array_previous = []
                            data_array_previous = data_array[min_last_pos_x_y_in_array+20:]
                            adjusted_indices_previous = adjusted_indices
                            b_previous_find_x = True
                            previous_find_x = included_x
                    else:
                        end_lst_file_found = True

                    if len(data_array_previous) <1:
                        end_lst_file_found = True

                # data_array_previous = np.empty(0, dtype=np.uint16)
                if nb_adc_not_found < 9: # RBS, RBS135, RBS150, Gamma20 et Gamma70 peuvent être éteins
                    for num_line_adc in array_adc:
                        if num_line_adc == 1 or num_line_adc == ADC_X or num_line_adc == ADC_Y or num_line_adc == 55: continue
                        adc2read = num_line_adc + 1
                        detector = ret_adc_name(num_line_adc)
                        if num_line_adc <= 4 :
                            data = cube_one_pass_pixe[num_line_adc]
                        elif num_line_adc == 5 or num_line_adc == 6 or num_line_adc == 7:
                            data = cube_one_pass_rbs[num_line_adc-5]
                        elif num_line_adc == 10:
                             data = cube_one_pass_gamma20[0]
                        elif num_line_adc == 11:
                            data = cube_one_pass_gamma70[0]

                        mysum1 = np.sum(data, axis=0)
                        mysum1 = np.sum(mysum1, axis=0)  # Somme des lignes
                        mysum1 = np.sum(mysum1, axis=0)  # Somme des lignes
                        if mysum1 > 0:
                            AGLAEfunction.feed_hdf5_map(data, path_lst, detector, num_pass_y,_dict_adc_metadata_arr[num_line_adc])
                        
                    AGLAEfunction.create_combined_pixe(cube_one_pass_pixe,path_lst,num_pass_y,_dict_adc_metadata_arr)
                    

                    print('\n')

    def ascii_to_hdf5(path_ascii:str, dict_metadata_global:dict,  dict_adc_metadata_arr:np):
        """Fonction pour lire les fichiers ascii spectra et les mettre dans un fichier HDF5"""
        head_tail = os.path.split(path_ascii)# Split le Path et le fichier
        datainname = head_tail[1].split("_")
        sum_all_x = [0,0,0,0,0]
    
        if len(datainname) > 4:
            _dateacq = datainname[0]
            _num_analyse = datainname[1]
            _objetacq = datainname[2]
            _projetacq = datainname[3]
        else:
            _dateacq = "?"
            _objetacq = "?"
            _projetacq = "?"
    
        type_hdf5 = dict_metadata_global["target type"] # OBJ ou STD
        hdf5_group = _dateacq + "_" + _num_analyse + "_" + dict_metadata_global["ref. analyse"] #_type_hdf5
        list_spectra_aglae , file_x0 = list_spectra_with_same_name(path_ascii) #" Liste des fichiers spectra .X0, .x1, .x2, .x3, .x4, ...."

        gupix_acq_time, real_dose = gupix_metadata_acq_time_and_dose(os.path.join(head_tail[0],file_x0[0])) #"""Récupère les métadonnées du fichier GUPIX"""
        dict_metadata_global["acquisition time (sec.)"] = gupix_acq_time
        dict_metadata_global["dose"] = real_dose
        dict_metadata_global.update(dict_metadata_global) # Met à jour les métadonnées globales avec les métadonnées GUPIX
            

        print(type_hdf5)
        #hdf5_file_obj = os.path.join(head_tail[0],dateacq + "_" +projetacq + "_" + 'type_hdf5' + ".hdf5")
        hdf5_file_obj = os.path.join(head_tail[0],_dateacq + "_" + _projetacq + "_" + "batch_IBA" + ".hdf5")
        hdf5_file_std = os.path.join(head_tail[0],_dateacq + "_" + _projetacq + "_" + "standard_IBA" + ".hdf5")


        if str.upper(type_hdf5) == 'STANDARD' or str.upper(type_hdf5) == 'STD': # OBJ + STD
            AGLAEfunction.create_empty_prj_hdf5(hdf5_file_std,hdf5_group)
            AGLAEfunction.metadata_prj_hdf5(hdf5_file_std,hdf5_group,dict_glob_metadata=dict_metadata_global)
            list_hdf5 = [hdf5_file_std]
        else: # only OBJ
            hdf5_file_std = "none" 
            #AGLAEFile.create_empty_prj_hdf5(hdf5_file_obj,hdf5_group,dict_glob_metadata=dict_metadata_global)
            AGLAEfunction.create_empty_prj_hdf5(hdf5_file_obj,hdf5_group)
            AGLAEfunction.metadata_prj_hdf5(hdf5_file_obj,hdf5_group,dict_glob_metadata=dict_metadata_global) 
            list_hdf5 = [hdf5_file_obj]

        metadata_one_adc = dict_adc_metadata_arr[0]
        list_ext = ["X1", "X2", "X3", "X4", "X0", "R8", "R135", "R150", "Coord_X", "Coord_Y", "G20", "G70","IL0"]
        list_ext_combined = ["X10", "X11", "X12", "X13", "X14", "X1234", "X34", "X23"]
        
        for spe in list_spectra_aglae: # list des fichiers ascii spectra .X0, .x1, .x2, .x3, .x4, ....
            _path_spe=os.path.join(head_tail[0],spe)
            str_ext_spe = str.upper(spe.split(".")[1]) #'Recupère extension du fichier'       
            
            try:    
                adc_indx= list_ext.index(str_ext_spe)
            except:
                adc_indx = -1
                #print("Not found in list_ext")
            
            if adc_indx == -1 : # Si pas trouvé dans la liste des spectres alors combined X'
                try:
                    metadata_one_adc={}
                    G_header = dict_gupix_metadata["gupix header"]
                    metadata_one_adc = build_metadata_combined(str_ext_spe,sum_all_x,G_header)
                                                
                except:
                    metadata_one_adc["Det. type"] = 'Extension not found'
                                            
            else:
                if adc_indx < 5:
                    metadata_one_adc = dict_adc_metadata_arr[adc_indx]
                    dict_gupix_metadata = gupix_metadata_from_path(_path_spe)
                    metadata_one_adc.update(dict_gupix_metadata)
                    sum_all_x[adc_indx] = int(metadata_one_adc["spectra sum"])
                else:
                    metadata_one_adc = {}
                    metadata_one_adc = dict_adc_metadata_arr[adc_indx]
                    #metadata_one_adc["acquisition time (sec.)"] = gupix_acq_time
                    #metadata_one_adc.update(dict_gupix_metadata)
                    
            try:
                np_data_spectre = read_ascii_spectra(_path_spe,dict_metadata_global)
            except:
                np_data_spectre = read_ascii_spectra_2column(_path_spe)

            for local_hdf5_file in list_hdf5:
                AGLAEfunction.feed_existing_hdf5(filename= local_hdf5_file, data= np_data_spectre, group_name= hdf5_group, dataset_name= str_ext_spe, metadata_one_adc= metadata_one_adc) 

    def copy_hdf5_to_folder(hdf5_file, path_ascii):
        """Copie un fichier HDF5 vers un dossier de destination."""
        head_tail = os.path.split(path_ascii)  # Split le Path et le fichier
        dest_folder = os.path.join(head_tail[0], "HDF5_maps_files")  
        dest_file = dest_folder + os.sep + os.path.basename(hdf5_file)  # Nom du fichier HDF5 sans le chemin

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        shutil.copy(hdf5_file, dest_file)

    @staticmethod
    def create_hdf5_filename(path_lst):
        """Crée le chemin du fichier HDF5 à partir du chemin du fichier LST."""
        head_tail = os.path.split(path_lst)  
        # Split le Path et le fichier
        destfile = head_tail[1].split(".")
        path_hdf5 = destfile[0] + ".hdf5"
        return path_hdf5

@staticmethod
def getSize(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size


@staticmethod
def get_x_to_exclude(croissant,columns,last_x_value,first_x_value,change_line,fin_lst):
    """Recupére X pour exclure dans ce process"""
    if columns==True and change_line==False and fin_lst==False:
        if croissant== True:
            find_x = last_x_value -1 #on va eclure le X-1
        elif croissant== False:
            find_x = first_x_value +1
    else:
        if croissant== True:
            find_x = last_x_value
        else:
            find_x = first_x_value
        
    return find_x

@staticmethod
def get_x_end_line_scan(croissant,sizex):
    """Get le X max du scan suivant ligne pair/impaire"""
    if croissant==True:
        end_x=sizex-1
    else:
        end_x=0
    return end_x

@staticmethod
def look_if_end_lst(max_val_y_lue,sizeY,val_x_fin_map,val_fin_x):
    """informe si on atteins la fin du fichier LST"""
    if max_val_y_lue==sizeY-1 and val_x_fin_map == val_fin_x: #Fin du fichier LST ?
        fin_lst = True
    else: 
        fin_lst = False
    return fin_lst

@staticmethod
def look_if_next_line(max_val_y_lue,y_scan_total):
    """informe si le dataset contient la fin du scan """
    
    if max_val_y_lue > y_scan_total: 
        change_line= True
    else:
        change_line =False

    return change_line
   
@staticmethod          
def ret_num_adc(detector):
   """Retourne le numéro ADC en binaire correspondant au nom du détecteur."""
   switcher = {
                "LE0":  0b0000000000010000, #2A
                "HE1":  0b0000000000000001,
                "HE2":  0b0000000000000010,
                "HE3":  0b0000000000000100,
                "HE4":  0b0000000000001000,
                "HE10": 0b0000000000001111,
                "HE11": 0b0000000000000011,
                "HE12": 0b0000000000001100,
                "HE13": 0b0000000000000111,
                "RBS" : 0b0000000010000000, #2D
                "GAMMA": 0b0000000000100000, #2B
                            }
   return switcher.get(detector)

@staticmethod
def get_indx_sum(str_ext_spe):
    """Retourne les indices à utiliser pour la somme des spectres combinés."""
    # Définition des variables représentant les indices à sélectionner
    switcher = {
                "X10":  [0, 1, 2, 3], #2A
                'X1234' : [0, 1, 2, 3],
                'X34' : [2, 3],
                'X23' : [1, 2],
                'X12' : [2, 3],
                'X13' : [0, 2],
                "X11":  [0, 1], #2A
                "X14":  [0, 3], #2A
                "X123": [0, 1, 2], #2A
                "X134": [0, 2, 3], #2A
                }
    return switcher.get(str_ext_spe)

@staticmethod
def sum_by_indices(arr, indices):
    """Somme les éléments d'un tableau selon les indices spécifiés."""
    sum1 =  sum(arr[i] for i in indices)
    return sum1

@staticmethod
def build_metadata_combined(str_ext_spe, sum_all_x,G_header):
    """Construit les métadonnées combinées à partir des données fournies."""
    metadata_one_adc = {}
    #indx_combined = list_ext_combined.index(str_ext_spe)
    indx_to_sum = get_indx_sum(str_ext_spe)
    sum_x2 = sum_by_indices(sum_all_x, indx_to_sum)

    # Remplace avec la bonne sum
    parts = G_header.split(' ')
    parts[4] = str(sum_x2)
    new_header = ' '.join(parts)
    metadata_one_adc["spectra sum"] = str(sum_x2)
    metadata_one_adc["gupix header"] = new_header
    metadata_one_adc["Det. type"] = "SDD Ketek"
    indices_str = " + ".join(f"X{i+1}" for i in indx_to_sum)
    metadata_one_adc["Combination of "] = f"detector {indices_str}" #str_ext_spe
    metadata_one_adc["Det. filter"] = "See Combination of"
    metadata_one_adc["Det. thickness"] = "See Combination of"

    return metadata_one_adc

@staticmethod
def ret_adc_name(num_adc):
   """"""
   switcher = {
               0: "X1",
               1: "X2",
               2: "X3",
               3: "X4",
               4: "X0",  # 2A
               5: "RBS",
               6: "RBS135",
               7: "RBS150",
               8: "Coord_X",
               9: "Coord_Y",
               10: "GAMMA20",
               11: "GAMMA70",
               12: "X12",
               13: "X13",
               14: "X14",
               23: "X23",
               134:"X134",
               34: "X34",
                            }
   return switcher.get(num_adc)

def ret_combined_name(ext_file_spe):
   """Retourne le nom de la combinaison des détecteurs"""
   switcher = {
               "X10": "X1+X2",
               "X11": "X1+X2",
               "X12": "X3+X4",
               "X13": "X1+X3",
               "X14": "X1+X4",
               "X23": "X2+X3",
               "X134":"X1+X3+X4",
               "X34": "X3+X4",
               "X1234": "X1+X2+X3+X4",
               "X123": "X1+X2+X3",
                            }
   return switcher.get(ext_file_spe)


@staticmethod
def ret_range_bytes(val):
    """Donne n°bits max pour valeur"""
    for bits in range(16):
        if val & (0b0000000000000001 << bits):
            nombre_bytes = bits
    return  2**(nombre_bytes+1) - 1

@staticmethod    
def resource_path(relative_path):
    """Obtenir le chemin absolu vers une ressource, fonctionne pour dev et build PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

@staticmethod    
def read_cfg_adc(path=None):
        if not path: path = 'config_lst2hdf5.ini'
        config = configparser.ConfigParser()
        config.read(resource_path(path))  # Utilise resource_path pour trouver le fichier de configuration
        config.sections()
        dict_config_mpawin_adc = {}
        dict_channel_adc ={}
        dict_combined_adc ={}
                
        nb_channels_pixe = config.getint('NB_CHANNEL', 'PIXE')
        nb_channels_rbs = config.getint('NB_CHANNEL', 'RBS')
        nb_channels_gamma20 = config.getint('NB_CHANNEL', 'GAMMA20')
        nb_channels_gamma70 = config.getint('NB_CHANNEL', 'GAMMA70')

        section_name = 'NB_CHANNEL'
        for key, value in config[section_name].items():
            print(f"{key}: {value}")
            dict_channel_adc[key] = value
        # Lecture des valeurs de la section MPAWIN
        
        section_name = 'CFG_MPAWIN'
        for key, value in config[section_name].items():
            print(f"{key}: {value}")
            dict_config_mpawin_adc[key] = value
        
        section_name = 'CFG_CONBINED'
        for key, value in config[section_name].items():
            print(f"{key}: {value}")
            dict_combined_adc['combined' + key] = value
            
        return dict_channel_adc,dict_config_mpawin_adc,dict_combined_adc

@staticmethod
def images_to_hdf5(path_ascii:str, dict_metadata_global:dict):
    '''Fonction pour lire les images du dossier Screen_capture et les mettre dans un fichier HDF5'''
    head_tail = os.path.split(path_ascii)# Split le Path et le fichier
    datainname = head_tail[1].split("_")

    if len(datainname) > 4:
        _dateacq = datainname[0]     #En général la date acquisition -> 20250105''
        _num_analyse = datainname[1] # Numéro de l'analyse
        _objetacq = datainname[2]    # Objet de l'acquisition
        _projetacq = datainname[3]   # Projet de l'acquisition
    else:
        _dateacq = "?"
        _objetacq = "?"
        _projetacq = "?"
    # dateacq =  dict_metadata_global["dateacq"] #En général la date acquisition -> 20250105''
    # num_analyse = dict_metadata_global["num analyse"]
    # projetacq = dict_metadata_global["prj"]

    _type_hdf5 =dict_metadata_global["target type"] # object ou standard

    _hdf5_group = _dateacq + "_" + _num_analyse + "_" + dict_metadata_global["ref. analyse"] #_type_hdf5
    _path_image_dir = os.path.join(head_tail[0], 'Screen_capture')
    path_image = os.path.join(_path_image_dir,_dateacq + "_" + _num_analyse)
    try: 
        list_image, image_video = get_images_by_base_name(path_image)
    except:
        list_image = []
        image_video = []
        print("No images found in Screen_capture folder")
        return  
    
    list_hdf5 = []
    hdf5_file_obj = os.path.join(head_tail[0],_dateacq + "_" +_projetacq + "_" + "batch_IBA" + ".hdf5")
    hdf5_file_std = os.path.join(head_tail[0],_dateacq + "_" +_projetacq + "_" + "standard_IBA" + ".hdf5")

    if str.upper(_type_hdf5) == 'STD' or str.upper(_type_hdf5) == 'STANDARD': # OBJ + STD   
        list_hdf5 = [hdf5_file_std]
        hdf5_file_obj = "none"
    else:
        list_hdf5 = [hdf5_file_obj]
        hdf5_file_std = "none"   

    for local_hdf5_file in list_hdf5:
        if  list_image != []:
            AGLAEfunction.feed_image_hdf5(filename=local_hdf5_file, image_path=list_image, 
                                    path_image_dir= _path_image_dir,group_name=_hdf5_group)   
        if image_video != []:
           img_name = _dateacq + "_" + _num_analyse + "_" + "image_area"
           AGLAEfunction.feed_image_video_hdf5(img_name=img_name,filename=local_hdf5_file, path_image_video=image_video, 
                                           path_image_dir= _path_image_dir, group_name=_hdf5_group)

 


                        
        # if type_hdf5 == "STD":

        #     AGLAEFile.feed_existing_hdf5(filename= hdf5_file_std, data= np_data_spectre, group_name=hdf5_group, dataset_name = hdf5_dataset)
        # else:
        #     AGLAEFile.feed_existing_hdf5(filename= hdf5_file_obj, data= np_data_spectre, group_name=hdf5_group, dataset_name = hdf5_dataset)
def search_Dose(header:str):
    """Search for the dose in the header string"""
   # str = "2025 06 55252 741 6065686  'KNHLNorth-1-02,1000,1000,50,50,500,40,2000000,Proton , 2999 keV , 40mm He , 50 um Al , OFF , 50 um Al + 20 um Cr , OFF'"
    par1 = header.split("'")
    par2 = par1[1].split(',')
    if len(par2) > 6:   
        dose = par2[7]  # Dose is the 7th element in the list
        dose = dose.strip()  # Remove any leading or trailing whitespace
    else:
        dose = "0"
    return dose

     
def read_ascii_spectra(path_ascii:str, dict_metadata_global:dict=None):
    """Read spectra format GUPIX -> 1 Column"""
    with open(path_ascii) as File_Spectre:
        header1 = File_Spectre.readline()
        header1 = File_Spectre.readline()
        data_spectre = File_Spectre.readlines()
        data_spectre = np.array(data_spectre)
        data_spectre = data_spectre.astype(np.int32)               
    return data_spectre

def read_ascii_spectra_2column(path_ascii:str):
    """Read spectra format SimNRA (RBS) -> 2 Column"""

    with open(path_ascii) as File_Spectre:
        header1 = File_Spectre.readline()
        as_header = False
        if header1 == "[DISPLAY]\n":
            as_header = True
            str = ""
            while str !="[DATA]\n":
                str = File_Spectre.readline()
                if str =="[DATA]\n":
                    break
                                
        data_spectre = File_Spectre.readlines()
        if as_header== False:
            data_spectre.insert(0,header1) 
        np_data_spectre = np.empty(len(data_spectre),dtype= np.int32)
        t= [line.split('\t') for line in data_spectre]
        for line,values in t:
            np_data_spectre[np.int32(line)]= np.int32(values)
    
    return np_data_spectre



def read_ascii_metadata(path_ascii_metafile:str):
    """Read metadata from GUPIX ASCII file"""
    t= 0
    dict_metadata_global = {}
    dict_adc_metadata_arr = np.full((20), {})
    dict_metadata = {}  
    end_adc = False
    indexadc = 0    
    adc_readed = np.array([0,0,0,0,0,0,0,0,0,0,0,0],dtype=bool)
    with open(path_ascii_metafile) as File_Metadata:
        all_metadata = File_Metadata.readlines()
        all_metadata = [line.strip() for line in all_metadata] # Remove whitespace

    det_aglae = ["X0", "X1", "X3", "X4", "X10", "X11", "X12", "X13", "RBS", "RBS150", "RBS135",
                     "GAMMA", "GAMMA70", "GAMMA20", "IBIL", "FORS"]
    
    fin_header =False

    for line in all_metadata:
        tmpheader = line.strip() # Remove whitespace
        if tmpheader == "": continue
        t+=1
        if t == 495:
                t=t
        else:
            t=t           
        # Map size:1280,1280,64,64,0,2564,100000_
        if "[ORS" in str.upper(tmpheader) or "[MAP" in str.upper(tmpheader): # Fin lecture ADC
            end_adc = True

        if "[ADC" in str.upper(tmpheader): # Read metadata ADC1
            mynumero = re.search(r'\[ADC(\d+)\]', tmpheader) #re.search(r'[ADC(\d+)]', tmpheader) 
            indexadc = int(mynumero.group(1)) -1
            dict_metadata.clear() #dict_metadata_arr[indexadc]
            idx_metadata = 0
            adc_readed[indexadc] = 1
    
        if indexadc == 0 and "REPORT-FILE from  written" in str(tmpheader): 
            para = str.split(str(tmpheader), sep='written')
            dict_metadata_global["timestamp"]= AGLAEfunction.clean_text(para[1])

        if "cmline1=" in str(tmpheader): # Nom detecteur
            para = str.split(str(tmpheader), sep='=')
            text_adc_name = AGLAEfunction.clean_text(para[1])
            text_adc_name= text_adc_name.upper()
            if text_adc_name in det_aglae:
                dict_metadata["adc name"]= text_adc_name
            else:
                dict_metadata["adc name"]= "det. OFF"

        if "DET:" in str.upper(tmpheader): # Info detecteur
            para = str.split(str(tmpheader), sep='=')
            para2 = str.split(para[1], sep=',') # 
            par_det= str.split(para2[0], sep=':')
            for i, text in enumerate(para2):
                text = AGLAEfunction.clean_text(text)
                text2 = str.split(text, sep=':')[1]
                if "DET:" in str.upper(text): 
                    dict_metadata["det. type"] = str.split(text, sep=':')[1]
                if "AREA :" in str.upper(text):
                    dict_metadata["det. active area"]= str.split(text, sep=':')[1]
                if "TCK" in str.upper(text):
                    dict_metadata["det. thickness"] =str.split(text, sep=':')[1]
                if "WIN:" in str.upper(text):
                    dict_metadata["det. entrance window"] =str.split(text, sep=':')[1]
                if "ANGLE:"in str.upper(text) :
                    dict_metadata["det. angle"] =str.split(text, sep=':')[1]
                if  "S/N:" in str.upper(text): 
                    dict_metadata["det. S/N"] =str.split(text, sep=':')[1]
        
        if "FILTER:" in str.upper(tmpheader): 
            para = str.split(str(tmpheader), sep='=')
            para2 = str.split(para[1], sep=',')
            #para = str.split(str(tmpheader), sep=':')
            #text = AGLAEFile.clean_text(para[1])  #remplace µ par u et supprime \\r\\n
            for i, text in enumerate(para2):
                text = AGLAEfunction.clean_text(text)
                #dict_metadata["filter"] =text
                if "WIN:" in str.upper(text):
                    dict_metadata["det. entrance window"] =str.split(text, sep=':')[1]
                if "ANGLE:"in str.upper(text) :
                    dict_metadata["det. angle"] =str.split(text, sep=':')[1]
                if "FILTER:"in str.upper(text) :
                    dict_metadata["det. filter"] =str.split(text, sep=':')[1]

        if "CALIBRATION:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])
            dict_metadata["calibration"] = text

        if "INSTITUTION:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])  #remplace µ par u et supprime \\r\\n
            dict_metadata_global["institution"] = text

        if "SAMP. INFO:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])
            dict_metadata_global["analyse description"] = text

        if "REF ANALYSE:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])
            # if "STANDARD:" in str.upper(text): #if text == "satndar":
            dict_metadata_global["ref. analyse"] = text
            # else:
            #    dict_metadata_global["ref. object"] = text
                

        if "USERNAME:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])  
            dict_metadata_global["username"] = text

        if "PRJ EUPHROSYNE:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])  
            dict_metadata_global["prj euphrosyne"] = text

        if "OBJ EUPHROSYNE:" in str.upper(tmpheader):
            para = str.split(str(tmpheader), sep=':')
            text = AGLAEfunction.clean_text(para[1])  
            dict_metadata_global["obj euphrosyne"] = text
            if str.upper (text) == "STANDARD":
                dict_metadata_global["target type"] = "standard"
            else:
                dict_metadata_global["target type"] = "object"

        if "MAP SIZE:" in str.upper(tmpheader):
            para2 = str.split(str(tmpheader), sep=':')
            para2 = str.split(para2[1], sep=",")
            map_info = ["map size x (um)","map size y (um)","pixel size x (um)","pixel size y (um)","pen size (um)","dose/column","dose" ]
            for i, text in enumerate(para2): 
                text = AGLAEfunction.clean_text(text)
                para2[i] = text
                dict_metadata_global[map_info[i]] = text
                
        if "EXP.INFO" in str.upper(tmpheader): 
            para2 = str.split(str(tmpheader), sep=':')
            para2 = str.split(para2[1], sep=",")
            beam_info = ["particle", "beam energy"]

            for i, text in enumerate(para2): 
                text = AGLAEfunction.clean_text(text)
                dict_metadata_global[beam_info[i]] = text
        
        if np.any(adc_readed) == True and end_adc == False:
            dict_adc_metadata_arr[indexadc] = dict_metadata.copy()
                
    return  dict_adc_metadata_arr, dict_metadata_global


def get_images_by_base_name(path_image):
    """Retourne les fichiers screen_capture  et capture video correspondant à un nom de base (ex:"20250225_0001*")"""
    head_tail = os.path.split(path_image)# Split le Path et le fichier
    destfile = head_tail[1].split(".")
    # Utilisez glob pour trouver les fichiers correspondants
    try:    
        fichiers = os.listdir(head_tail[0])
    except:
        print("No such file or directory")
        return  None
        
    # Filtre les fichiers avec le motif "toto.*"
    fichiers_image = fnmatch.filter(fichiers, destfile[0] +'*.png')
    fichiers_image_video = fnmatch.filter(fichiers, destfile[0] +'*Video.jpg')
    
    
    return fichiers_image,fichiers_image_video

def list_spectra_with_same_name(path_ascii):
    """Liste les fichiers spectres avec le même nom que le fichier ASCII"""
    head_tail = os.path.split(path_ascii)# Split le Path et le fichier
    destfile = head_tail[1].split(".")
    fichiers = os.listdir(head_tail[0])
    # Filtre les fichiers avec le motif "destfile"
    file_X0 = fnmatch.filter(fichiers, destfile[0] +'.X0')
    fichiers_spectra = fnmatch.filter(fichiers, destfile[0] +'.X0')
    for ext in ['X1', 'X3', 'X4', 'X11','X23', 'X134', 'X34', 'X13', 'X10','X1234']:
        fichiers_spectra += fnmatch.filter(fichiers, destfile[0] +'.' + ext)
    fichiers_spectra += fnmatch.filter(fichiers, destfile[0] +'.R*')
    fichiers_spectra += fnmatch.filter(fichiers, destfile[0] +'.G*')
    fichiers_spectra += fnmatch.filter(fichiers, destfile[0] +'.IL*')
  
    return fichiers_spectra , file_X0

def gupix_metadata_from_path(path_ascii):
    """Récupère les métadonnées du fichier GUPIX à partir du chemin du fichier ASCII"""
    dict_metadata = {}
    with open(path_ascii) as f:
       header1 = f.readline()
       header1 = f.readline()
       gupix_hearder= header1.split("'")
       dict_metadata["gupix header"] = gupix_hearder[0] 
       gupix_hearder2= header1.split(" ")
       dict_metadata["spectra sum"] = gupix_hearder2[4]

    return dict_metadata

def gupix_metadata_acq_time_and_dose(path_ascii):
    """Récupère le temps d'acquisition du fichier GUPIX"""
    acq_time = "0"
    with open(path_ascii) as f:
       header1 = f.readline()
       header1 = f.readline()
       gupix_hearder= header1.split("'")
       gupix_hearder2= header1.split(" ")
       acq_time = gupix_hearder2[3]
       try:
        real_dose= search_Dose(header1)
        real_dose = real_dose.replace("'", "")
       except:
        real_dose = "0" # Default dose if not found in header"

    return acq_time,real_dose


def main():
    print("hello master")
    
    _path_lst = ""
    _fnct = "maps"
    _path_ascii = ""
    _ibil = "" 
    if len(sys.argv) < 2:
        print("Usage:  <arg1:Path of LST file> <arg2: type of extraction 'maps' or 'spectra'> <arg3:path of one ASCII spectra>")
        #_path_lst = 'C:\\Data\\2025\\Lst_2025\\20250128_0015_OBJ_IMAGERIE_IBA.lst'
       # _path_lst = 'C:\\Data\\2025\\Lst_2025\\20250128_0015_OBJ_IMAGERIE_IBA.lst'
        #_path_lst ="C:\\Data\\2025\\Lst_2025\\20250206_0008_OBJ_AGLAE_IBA.lst"
        #_path_lst ="C:\Data\\2025\\lst_FX\\20250519_0181_OBJ_BOS_IBA.lst"
        #_path_lst ="C:\\Data\\2025\\lst_CavePit\\20250616_0016_KNHLNorth-1_CAVEPIT_IBA.lst"
        _path_lst ="C:\\Data\\2025\\fff\\20250618_0001_STD_AGR_IBA.lst"
        _path_lst = "C:\\Data\\2025\\2025_loisel\\LST\\lst\\20250626_0048_peinture_FR-STAINEDGLASS_IBA.lst"
        _path_lst = "C:\\Data\\2025\\Yvan_IBIL\\lst\\20250630_0029_OBJ_SRV-VISHNU_IBA.lst"
        _path_ascii = "NULL"
        #_path_ascii = "C:\\Data\\20220725_Renne-provenance\\20220725_Renne-provenance\\20220725_0001_std_RENNE-PROV_IBA.x0"
        #_path_ascii ="C:\\Data\\2025\lst_FX\\20250519_0181_OBJ_BOS_IBA.x0"
        #_path_ascii ="C:\\Data\\2025\\lst_CavePit\\20250616_0016_KNHLNorth-1_CAVEPIT_IBA.x0"
        _path_ascii = "C:\\Data\\2025\\fff\\20250618_0001_STD_AGR_IBA.x0"
        _path_ascii = "C:\\Data\\2025\\Yvan_IBIL\\20250630_SRV-Vishnu\\20250630_0001_STD_SRV-VISHNU_IBA.x0"
        _fnct = "MAPS"
        #_fnct = "SPECTRA"
    else:
        _fnct = str.upper(lst_arg[1])
        _path_lst = lst_arg[2]

        if len(sys.argv) > 2: # En cas de cas MAPS pas d'argument 3
            _path_ascii = lst_arg[3]

        #return
    #_fnct =str.upper("spectra")
    if _fnct not in ["MAPS", "SPECTRA"]:
        print("Function must be 'MAPS' or 'SPECTRA'")
        return
   
    head_tail = os.path.split(_path_lst) # Split le Path et le fichier
    #_path_metafile = os.path.join(head_tail[0], head_tail[1].split(".")[0] + ".metafile") # Chemin du fichier de métadonnées
    #dict_adc_metadata_arr,dict_metadata_global = read_ascii_metadata(_path_metafile) # Lecture des métadonnées du fichier ASCII spectra
    dict_adc_metadata_arr,dict_metadata_global = AGLAEfunction.open_header_lst(_path_lst)
    path_hdf5 = os.path.join(head_tail[0], head_tail[1].split(".")[0] + ".hdf5") # Chemin du fichier HDF5
    hdf5_filename =AGLAEfunction.create_hdf5_filename(_path_lst) # Crée le nom du fichier HDF5 à partir du nom du fichier LST
    
    for arg in sys.argv[1:]:
        print(f"Argument: {arg}")

    if _fnct== "MAPS":
        #_config_adc = read_cfg_adc()
        AGLAEfunction.extract_lst_vector(_path_lst, dict_metadata_global, dict_adc_metadata_arr)
        AGLAEfunction.write_hdf5_metadata(_path_lst, dict_metadata_global)
        ibil = False
        ibil,path_edf = AGLAEfunction.search_ibil_folder(_path_ascii)
        if ibil == True:
            AGLAEfunction.ibil_To_hdf5(path_edf,"IBIL",_path_lst,dict_adc_metadata_arr)

        AGLAEfunction.copy_hdf5_to_folder(path_hdf5,_path_ascii)
    elif _fnct == "SPECTRA":
        AGLAEfunction.ascii_to_hdf5(_path_ascii,dict_metadata_global,dict_adc_metadata_arr)
        images_to_hdf5(_path_ascii,dict_metadata_global)

if __name__ == "__main__":
   lst_arg = sys.argv
   main()
