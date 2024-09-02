import h5py, re
import numpy as np
import sys, os
from PyQt5.QtCore import  QThread, pyqtSignal
import threading
from datetime import datetime
from time import perf_counter
import matplotlib.pyplot as plt


class ThreadReadLst(QThread):

    valueChanged = pyqtSignal(int)

    def __init__(self,path):
        self.path = path
        self.detector = "LE0"

    # Create a counter thread


    def run(self):

        # pathlst = "C:\\Dev\\PyPIX\\Data\\26jul0068.lst" #26jul0068.lst
        pathlst1 = self.path
        tmpheader = ""
       # MyResultat = list()
        header1 = list()
        sizeX = 1
        sizeY = 1
        print("toto")
        print(pathlst1)
        fin_header = False
        t=0
        with open(pathlst1, "rb") as file_lst:

            while fin_header == False: #tmpheader != b'[LISTDATA]\r\n':
                tmpheader = file_lst.readline()
                t+=1
                # Map size:1280,1280,64,64,0,2564,100000
                if "Map size" in str(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    para = str.split(para[1], sep=",")
                header1.append(tmpheader)
                fin_header = tmpheader == b'[LISTDATA]\r\n' or tmpheader == b'[LISTDATA]\n' or t > 5000
                
            print(header1)
            print(para[0], para[1], para[2], para[3])
            sizeX = int(para[0]) / int(para[2])
            sizeY = int(para[1]) / int(para[3])
            sizeX = int(sizeX)
            sizeY = int(sizeY)
            print(sizeY)
            print(sizeX)
            adcnum = []
            cube = np.zeros((sizeX, sizeY, 2048), 'u4')

            # for i in range (0,50):
            val = b'\xff\xff\xff\xff'
            lstcontent = file_lst.read()
            ind1 = 0
            nrows = 0
            ncolumns = 0

            # MainPage.progress.setRange(1,len(lstcontent) -22000000)

            while ind1 < len(lstcontent) - 22000000:

                try:
                    # val = file_lst.read(4)
                    val = lstcontent[ind1:ind1 + 4]
                    ind1 += 4
                except:
                    val = b'\xff\xff\xff\xff'
                    # QtCore.QCoreApplication.processEvents()
                    # MainPage.progress.setValue(ind1)

                if val == b'\xff\xff\xff\xff':
                    # val = file_lst.read(4)
                    val = lstcontent[ind1:ind1 + 4]
                    ind1 += 4
                    text = "Events"
                    self.valueChanged.emit(int(100/len(lstcontent)*ind1))

                val3 = int.from_bytes(val, byteorder='little', signed=False)
                low1 = val3 & 0xFFFF
                hight1 = int(val3 >> 16)

                if 0x4000 == hight1:
                    text = "Valeur tempo"
                    tempo = low1
                if 0xFFFF == low1:
                    text = "Valeur channel"
                    channel = hight1

                if 0x8000 == hight1:
                    text = "TAG ADC"
                    adc = low1
                    adcnum = []
                    channel_num = []
                    for bits in range(8):
                        if adc & (0b00000001 << bits):
                            adcnum.append(bits)
                    if len(adcnum) % 2 == 1:  # & len(adcnum) > 1:
                        # toto = file_lst.read(2) #Nb croissa nte ADC !!
                        val = lstcontent[ind1:ind1 + 2]
                        ind1 += 2

                    for f in adcnum:
                        val_lue = lstcontent[ind1:ind1 + 2]
                        ind1 += 2
                        # val_lue = file_lst.read(2)

                        if f != 4 and f != 5: channel_num.append(int.from_bytes(val_lue, byteorder='little', signed=False))
                        if f == 4: nrows = int.from_bytes(val_lue, byteorder='little', signed=False)  # Valeur X
                        if f == 5: ncolumns = int.from_bytes(val_lue, byteorder='little', signed=False)
                    for c in channel_num:
                        if (c < 2048) & (nrows < 20) & (ncolumns < 20):
                            cube[nrows, ncolumns, c] += 1

            AGLAEFile.write_hdf5(cube, self.path, self.detector)


class AGLAEFile(object):
    _FILE_LOCK_1 = threading.Lock()

    def __init__(self):
        self.path = "c:/temp/toto.lst"
        self.detector = "LE0"

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
    def create_empty_hdf5(filename: str, data_shape, dtype=np.float64, group_name="XRF_analysis",
                          dataset_name="dataset", file_lock=_FILE_LOCK_1):
        with file_lock:
            with h5py.File(filename, 'w') as h5file:
                subgroup = h5file.require_group(group_name)
                dset = subgroup.create_dataset(dataset_name, shape=data_shape, dtype=dtype)

    @staticmethod
    def feed_existing_hdf5(filename, data, group_name="XRF_analysis", dataset_name="dataset", file_lock=_FILE_LOCK_1):

        with file_lock:
            with h5py.File(filename, 'a') as h5file:
                dset = h5file[f'{group_name}/{dataset_name}']
                dset[:] = data

    @staticmethod
    def get_dataset_data_hdf5(filename, group_name: str = "XRF_analysis", dataset_name: str = "dataset",
                              file_lock=_FILE_LOCK_1):

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
    def feed_hdf5_map(mydata, Pathlst, detector, FinalHDF, num_det,sizeX,sizeY,nbcanaux,num_scan_y):
     
        destfile = Pathlst.split(".")
        newdestfile = destfile[0] + ".hdf5"

        if destfile[1] == 'lst':
            newdestfile = destfile[0] + ".hdf5"
        elif destfile[1] == 'edf':

            newdestfile = os.path.join(os.path.dirname(Pathlst), FinalHDF)
    
        with h5py.File(newdestfile, 'a') as h5file:
            group_name = 'data' + str(num_det) + "_" + detector  # "data"

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
              


        h5file.close()
  
  

    def write_hdf5_metadata(Pathfile,parametre,detname,FinalHDF):
        # f = h5py.File('./Data/ReadLst_GZIP.hdf5', 'w')
        head_tail = os.path.split(Pathfile)# Split le Path et le fichier
        destfile = head_tail[1].split(".")
        newdestfile = destfile[0] + ".hdf5"
        index_iba= destfile[0].find("_IBA_")
        index_l1 = destfile[0].find("_L1_")
        index_xrf = destfile[0].find("_XRF1_:")
        det_aglae = ["LE0", "HE1", "HE2", "HE3", "HE4", "HE10", "HE11","HE12","HE13","RBS","RBS150","RBS135","GAMMA","GAMMA70","GAMMA20","IBIL","FORS"]
        iba_para = False

        for det1 in det_aglae:
            if detname == det1:
                iba_para = True



        if destfile[1] == 'lst':
            newdestfile = destfile[0] + ".hdf5"
      
        newdestfile1 =  os.path.join(head_tail[0] , FinalHDF)

        # print(newdestfile)
        try:
            f = h5py.File(newdestfile1, 'a')

        except:
             f = h5py.File(newdestfile1, 'w')
        try:
            del f["parametres"]
        except:
            pass
        try:
            del f["stack/detector"]
        except:
            pass

        if iba_para == True:
            grp = f.create_group("parametres")
            grp.attrs["Date"] = parametre[0]
            grp.attrs["Projet"] = parametre[1]
            grp.attrs["Objet"] = parametre[2]
            grp.attrs["Particule"] = parametre[8]
            grp.attrs["Beam energy"] = parametre[9]
            grp.attrs["Map size X/Y (um)"] = '{} x {}'.format(parametre[3], parametre[4])
            grp.attrs["Pixel size X/Y (um)"] = "{} x {} ".format(parametre[5], parametre[6])
            grp.attrs["Pen size (um)"] = "{} ".format(parametre[7])
            #(\u03BC) code mu
            grp.attrs["Detector filter"] = "LE0:{}, HE1:{}, HE2:{}, HE3:{}, HE4:{}".format(parametre[10], parametre[11],
                                                                                       parametre[12], parametre[13],
                                                                                       parametre[14])
            #print("HDF5 MetaData write")
        else:
          #  grp1 = f.create_group("stack/detector")
            #grp1.attrs["Data"] = "Test"
            grp = f.create_group("parametres")
            grp.attrs["Date"] = parametre[0]
            grp.attrs["Objet"] = parametre[1]

        f.close()


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
            newdestfile = destfile[0] + ".hdf5"
        elif destfile[1] == 'edf':
            # n = len(destfile[0])
            name = os.path.basename(destfile[0])
            if index_iba > 0:
                Myname = name.split('_IBA')
                FinalHDF = Myname[0] + ".hdf5"
            else:
                Myname = name.split('_')
                FinalHDF = Myname[0] + "_" + Myname[1] + ".hdf5"

        return FinalHDF


     
    def open_header_lst(pathlst):
        import os
        # pathlst = "E:/21mai0106.lst"
        tmpheader = ""
        para2 = ""
        header1 = list()
        sizeX = 1
        sizeY = 1
        head_tail = os.path.split(pathlst)  # Split le Path et le fichier
        root_text = os.path.splitext(head_tail[1])  # Split le fichier et ext

        datainname = root_text[0].split("_")
        if len(datainname) > 4:
            dateacq = datainname[0]
            objetacq = datainname[2]
            projetacq = datainname[3]
        else:
            dateacq = "?"
            objetacq = "?"
            projetacq = "?"

        header1.append(dateacq)
        header1.append(objetacq)
        header1.append(projetacq)
        t =0    
        with open(pathlst, "rb") as file_lst:
            import os
            size_lst = os.path.getsize(pathlst)
            fin_header =False
            while fin_header ==False: # != b'[LISTDATA]\r\n' or tmpheader != b'[LISTDATA]\n':
                tmpheader = file_lst.readline()
                t+=1
                if t == 615:
                     t=t
                else:
                    t=t           
                # Map size:1280,1280,64,64,0,2564,100000_
                if "Map size" in str(tmpheader):
                    para = str.split(str(tmpheader), sep=':')
                    para = str.split(para[1], sep=",")
                    for newwpara in para:
                        newwpara = newwpara.replace("\\n", '')
                        newwpara = newwpara.replace("'", '')
                        if len(header1) < 8:
                            header1.append(newwpara)
                        

                if "Exp.Info" in str(tmpheader):
                    para2 = str.split(str(tmpheader), sep=':')
                    para2 = str.split(para2[1], sep=",")

                    for i, text in enumerate(para2):  # remplace le code \\xb5 mu par la lettre u pour um
                        text = text.replace("\\xb5", 'u')
                        text = text.replace("\\r\\n", '')
                        text = text.replace("\\", 'jj')
                        para2[i] = text
                        header1.append(text)

                    para = para + para2
                fin_header = tmpheader == b'[LISTDATA]\r\n' or tmpheader == b'[LISTDATA]\n' or t > 5000
                # header1.append(tmpheader)
        if len(para2) == 0:
            for i in range(7):
                header1.append("?")

        return header1

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

    def return_index_adc_in_data_array(adjusted_indices,adc_values,num_line_adc):
        """ Retourne tout les indices des ADC dans DataArray"""
        
        # Op�rations bitwise et filtrage des indices
        adc_masked = np.bitwise_and(adc_values[:], 0b0000000000000001 << num_line_adc)
        coord_X_masket = np.bitwise_and(adc_values[:], 0b0000000000000001 << 8)
        coord_Y_masket = np.bitwise_and(adc_values[:], 0b0000000000000001 << 9)
        condition = (adc_masked != 0) # and coord_X_masket[:] != 0 and coord_Y_masket[:] != 0
        conditionX = coord_X_masket[:] != 0
        conditionY = coord_Y_masket[:] != 0
        condition2 = np.logical_and(condition, conditionX)
        condition_final = np.logical_and(condition2, conditionY)
     #   filtered_indices = np.where(adc_masked[:] != 0 and coord_X_masket[:] != 0 and coord_Y_masket[:] != 0, adjusted_indices[:],
       #                             np.full(len(adc_masked), -1, dtype=np.int16))
        filtered_indices = np.where(condition_final, adjusted_indices[:],
                                     np.full(len(condition_final), -1, dtype=np.int16))
        # filtered_indices = np.where(coord_Y_masket[:] != 0, adjusted_indices[:],
        #                             np.full(len(coord_Y_masket), -1, dtype=np.int16))

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
        try:
            first_x_value = int(first_x[0])
        except:
            first_x_value = int(first_x[0][0]) # plusieurs valeur superieur X identique 

        # if croissant == True:
        #     last_x_value = last_x - 1  # int(last_x[0]) - 1
        # else:
        #     if last_x != 0:
        #         last_x_value = first_x_value  # int(last_x[0]) + 1
        #     else:
        #         last_x_value = 0  # fin de la ligne de retour

        return first_x_value, last_x_value

      

    def read_indice_max_x(croissant,sizeX,coord_x,find_x):
 
            indice_x_max = np.where(coord_x == find_x)
            len_x_max = len(np.shape(indice_x_max[0]))
            if len_x_max < 1: #Pas ce X dans le Dataarray , faisceau OFF ?"

                while len_x_max < 1:
                    if croissant == True:
                        find_x -=1
                    else:
                        find_x +=1
                    indice_x_max = np.where(coord_x == find_x)
                    len_x_max = len(np.shape(indice_x_max[0]))    


            if croissant == True:
                try:
                    indice_x_last = indice_x_max[0][-1]
                except:
                    indice_x_last = indice_x_max[-1]
            else:
                try:
                    indice_x_last = indice_x_max[0][0]
                except:
                    indice_x_last = indice_x_max[0]
           
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

# Fonction d'extraction du fichier LST avec vectorisation
    def extract_lst_vector(path_lst, para,ADC_X, ADC_Y):
        pathlst1 = path_lst
        tmpheader = ""
        header1 = list()
        sizeX = 1
        sizeY = 1
        #print(path_lst)
        print("map size = ",para[3], para[4])
        sizeX = int(para[3]) / int(para[5])
        sizeY = int(para[4]) / int(para[6])
        sizeX = int(sizeX)
        sizeY = int(sizeY)
        adcnum = []

        nbcanaux = 1024
        nbcanaux_pixe = 2048
        nbcanaux_gamma = 2048
        nbcanaux_rbs = 512
        
        file = open(path_lst, 'rb')
        size_lst = getSize(file)
        file.close()
        allheader = ""
        fin_header = False
        t=0
        with open(path_lst, "rb") as file_lst:  # Trop long
            while fin_header == False: #tmpheader != b'[LISTDATA]\r\n':
                tmpheader = file_lst.readline()
                tmp1 = str(tmpheader)
                allheader = allheader + tmp1.replace("\\r\\n", '')
                t+=1    
                size_lst -= len(tmp1) - 2
                if "condition" in str(tmpheader):
                    toto = 1
                fin_header = tmpheader == b'[LISTDATA]\r\n' or tmpheader == b'[LISTDATA]\n' or t > 5000
              
            pensize = int(para[7])
            nb_pass_y = int(sizeY / (pensize / int(para[6])))
            nb_column_total = sizeX*nb_pass_y
           
                 # Pas possible gros LST
      
            range_very_small = 3*10 ** 6 # 3Mo
            range_small = 10*10 ** 6  # range(1, 10**6) #50 Mo
            range_50mega = 50*10 ** 6  # #100 Mo
            range_100mega = 100*10 ** 6  # #100 Mo
            range_300mega = 300*10 ** 6  # #100 Mo
            range_500mega = 500*10 ** 6  # #100 Mo

            
            range_giga = 1000 *10** 6  # range(10**7 + 1, 1**12)  # 100 Mo
            size_lst = int(size_lst / 2)  # car on lit des Uint16 donc 2 fois moins que le nombre de bytes (Uint8)
            size_block = size_lst
            size_one_scan = size_lst / nb_pass_y
            size_4_column_scan = size_one_scan / (sizeX/4)  # taille 4 column
            if size_4_column_scan < 10*10**6 and sizeX > 20:
                size_4_column_scan = size_one_scan / (sizeX/8) # taille 8 column 
            size_block = int(size_4_column_scan)

        
            
            if nb_pass_y == 1 and size_lst < 50*10**6: 
                size_block = size_lst
          
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

            if nb_pass_y % 2 == 0:
                last_x_maps = 0
            else:
                last_x_maps = sizeX - 1

            for num_pass_y in range(nb_pass_y):

                if (num_pass_y % 2 == 0):
                    croissant = True
                    next_x_value = np.zeros(12, dtype=np.uint16)

                else:
                    croissant = False
                    next_x_value = np.full(12, sizeX - 1, dtype=np.uint16)


                end_lst_file_found = False
                y_max_current_scan = y_scan + (num_pass_y * nb_column)

                cube_one_pass_pixe = np.empty((5, nb_column, sizeX, nbcanaux_pixe), dtype=np.uint32)
                cube_one_pass_gamma = np.empty((2, nb_column, sizeX, nbcanaux_gamma), dtype=np.uint32)
                cube_one_pass_rbs = np.empty((3, nb_column, sizeX, nbcanaux_rbs), dtype=np.uint32)

                fin_ligne = False
                change_line= False

                while (fin_ligne == False and end_lst_file_found == False):  # max_val_y_lue <= y_scan): # or croissa nte == False ):

                    adc2read = 0
                    adc_values = np.empty(0, dtype=np.uint16)
                    data_array = np.empty(0, dtype=np.uint16)
                    adjusted_indices = np.empty(0, dtype=np.uint16)
                    adjusted_indices_previous = np.empty(0, dtype=np.uint16)

                    min_last_pos_x_y_in_array = 0 #nb_byte_to_read
                    data_array = np.fromfile(file_lst, dtype=np.uint16, count=int(nb_byte_to_read))
                    if len(data_array) < nb_byte_to_read:
                        end_lst_file_found = True
                        print("End LST file found")

                    adjusted_indices,data_array ,shape_data_array = AGLAEFile.return_adc_adjusted_index (data_array_previous, data_array)
                    adc_values = np.array(data_array[adjusted_indices])


                    nb_read_total += (nb_byte_to_read * 2) + len(data_array_previous)
                    t1 = perf_counter()

                    array_adc = [0,1,2,3,4,6,7,10,11]
                    #array_adc = [0,4]
                    for num_line_adc in array_adc: #range(12):
                        if num_line_adc == 1 or num_line_adc == 8 or num_line_adc == 9 or num_line_adc == 5: continue

                        switcher = {5: nbcanaux_pixe, 0: nbcanaux_pixe, 1: nbcanaux_pixe, 2: nbcanaux_pixe, 3: nbcanaux_pixe, 4: nbcanaux_pixe, 80: nbcanaux_pixe, 81: nbcanaux_pixe,
                                    82: nbcanaux_pixe, 6: nbcanaux_rbs, 7: nbcanaux_rbs, 10: nbcanaux_gamma, 11: nbcanaux_gamma}
                        nbcanaux = switcher.get(num_line_adc)

                        detector = ret_adc_name(num_line_adc)
                        adc2read = num_line_adc + 1
                        # adc2read = ret_num_adc(self.detector)
                        t0 = perf_counter()
                        # Return 
                        non_zero_indices = AGLAEFile.return_index_adc_in_data_array(adjusted_indices,adc_values,num_line_adc)
                        if non_zero_indices[0] == -1:
                            continue
                        adc_words = data_array[non_zero_indices]
                        indice_val_to_read = AGLAEFile.return_val_to_read(adc_words,non_zero_indices)

                        max_size_x = ret_range_bytes(sizeX - 1)
                        max_size_y = ret_range_bytes(sizeY - 1)
                        coord_x = data_array[indice_val_to_read[ADC_X, :]]  
                        coord_x = coord_x & max_size_x  # & binaire pour masquer bits > max_size à 0
                        coord_y = data_array[indice_val_to_read[ADC_Y, :]]  
                        c1 = indice_val_to_read[9, :]
                        c1 = c1[c1 != 0]
                        
                        if len(c1) < 100:
                            continue
                        coord_y = coord_y & max_size_y 

                        # Met des -1 aux coord X et Y > valeur de la carto
                        out_range_x = np.where(coord_x > sizeX - 1)
                        coord_x = np.delete(coord_x, out_range_x)
                        coord_y = np.delete(coord_y, out_range_x)
                        coord_x = np.where(coord_x <= sizeX - 1, coord_x, np.full(len(coord_x), 0))
                        coord_y = np.where(coord_y <= sizeY - 1, coord_y, np.full(len(coord_y), 0))
                    
                        max_val_y_lue,min_val_y_lue = AGLAEFile.read_min_max_y(coord_y)
                        #first_x_value, last_x_value = AGLAEFile.read_range_x(coord_x, croissant)
                        first_x_value, last_x_value = AGLAEFile.range_x(coord_x, croissant)

                        
                        change_line = look_if_next_line(max_val_y_lue,y_max_current_scan) #True or False si Changement de Y sup.
                        val_x_fin_map = get_x_end_line_scan(croissant,sizeX) # retourne val 
                        
                        if change_line == False:
                            fin_lst = look_if_end_lst(max_val_y_lue,sizeY,val_x_fin_map,last_x_value)
                        else:
                            fin_lst = False
                        
                        find_x = get_x_to_exclude(croissant, columns, last_x_value,first_x_value,change_line,fin_lst)
                       
                        if change_line==True or fin_lst== True or end_lst_file_found == True: # Cas changement ligne ou fin fichier LST
                            if end_lst_file_found == True:
                                toto=1
                            
                            if last_x_value < first_x_value and croissant==True:
                                last_x_value = sizeX-1
                            
                            if fin_lst==False and end_lst_file_found == False:
                                indice_last = AGLAEFile.read_max_indice_change_colonne(coord_y,y_max_current_scan) #Recherche last_indice avec Y < scan total
                            elif end_lst_file_found == True or fin_lst == True:
                                indice_last = len(coord_y) -1

                            fin_ligne = True
                            if num_line_adc== 0 :
                                if croissant==True:
                                    print("X:", last_x_value,end=",")
                                else:
                                    print("X:",first_x_value,end=",")

                            coord_x = coord_x[:indice_last]
                            coord_y = coord_y[:indice_last]
                            max_data_array = indice_val_to_read[ADC_X, indice_last]

                            if max_data_array > min_last_pos_x_y_in_array:
                                  min_last_pos_x_y_in_array = max_data_array

                        else:  # recherche la dernire valeur de X
                            columns= get_colums_range(croissant,first_x_value,last_x_value)
                            if end_lst_file_found == False:
                                columns= get_colums_range(croissant,first_x_value,last_x_value)
                                if columns == True: # plus de 1 colonne
                                    indice_last = AGLAEFile.read_indice_max_x(croissant,sizeX,coord_x,find_x)#,next_x_value[num_line_adc])
                                else:
                                    find_x = first_x_value
                                    indice_last = AGLAEFile.read_indice_max_x(croissant,sizeX,coord_x,find_x)#,next_x_value[num_line_adc])
                            
                                
                                if num_line_adc== 0 :
                                    print("X:", last_x_value,end=",")
                                
                                max_data_array = indice_val_to_read[ADC_X, indice_last]
                                coord_x = coord_x[:indice_last]
                                coord_y = coord_y[:indice_last]


                                if max_data_array > min_last_pos_x_y_in_array:
                                        min_last_pos_x_y_in_array = max_data_array

                            else: # Fin du fichier on mets les bornes max pour X
                                if num_line_adc== 0 :
                                    print("X:", last_x_value,end=",")
                                if croissant == True:
                                    first_x_value, last_x_value = AGLAEFile.read_range_x(coord_x, croissant)
                                else:
                                    last_x_value = 0


                            indice_x_last = len(coord_x)


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
                        if (croissant == True and find_x==sizeX-1) or (croissant == False and find_x == 0):
                            fin_ligne = True
                        
                       
                        p1 = first_x_value
                        
                        if last_x_value == sizeX-1:
                            p2 = last_x_value # Je prend la dernier column en compte dans mon histogramme
                        elif croissant == False:
                            p2 = last_x_value
                        else:
                            p2 = last_x_value -1

                       
                        if croissant == True:
                            adc3 =adc1[0]
                            del adc1
                         
                            if columns == False:
                                # range_histo = 1first_x_value == last_x_value - 1 and fin_ligne == False: # Une seule column dans le dataArray
                                range_histo = 1
                            else:
                                r1 = [p1, p2]
                                range_histo = (p2 - p1) + 1

                        else:
                            new_coord_x = np.delete(new_coord_x, 0)
                            new_coord_x = np.flip(new_coord_x)
                            new_coord_y = np.delete(new_coord_y, 0)
                            new_coord_y = np.flip(new_coord_y)
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
                       
                        if range_histo==1:
                           H1, xedges, yedges= np.histogram2d(new_coord_y,adc3,bins=(nb_column,nbcanaux),range= ({0, nb_column-1},{0, nbcanaux-1}))
                        
                        else:
                            H1, edges = np.histogramdd((new_coord_y, new_coord_x, adc3),
                                                   range=({0, nb_column-1}, r1, {0, nbcanaux-1}),
                                                   bins=(nb_column, range_histo, nbcanaux))
                       
                    
                       
                        if croissant == True:
                            if last_x_value == sizeX-1:
                                last_x_value = last_x_value +1 # Incrément de 1 pour la derniere column car H1 a une dimension +1
                        else:
                            last_x_value = last_x_value +1


                        if num_line_adc <=4:
                            if range_histo == 1:
                                cube_one_pass_pixe[num_line_adc ,:, first_x_value, :] = H1
                            else:
                                cube_one_pass_pixe[num_line_adc][0:,first_x_value:last_x_value, 0:] = H1

                        elif num_line_adc == 6 or  num_line_adc == 7:
                            if range_histo == 1:
                                cube_one_pass_rbs[num_line_adc - 6][0:, int(next_x_value[num_line_adc]),0:] = H1
                            else:
                                cube_one_pass_rbs[num_line_adc - 6][0:,first_x_value:last_x_value, 0:] = H1

                        elif num_line_adc == 10 or num_line_adc == 11:
                            if range_histo == 1:
                                cube_one_pass_gamma[num_line_adc - 10][0:, int(next_x_value[num_line_adc]),0:] = H1
                            else:
                                cube_one_pass_gamma[num_line_adc - 10][0:,first_x_value:last_x_value, 0:] = H1

                    
                        if range_histo != 1 and croissant == True:
                            next_x_value[num_line_adc] = last_x_value
                        else:
                            next_x_value[num_line_adc] = first_x_value
                        

                    if min_last_pos_x_y_in_array < int(shape_data_array):
                        data_array_previous = []
                        data_array_previous = data_array[min_last_pos_x_y_in_array+5:]
                        adjusted_indices_previous = adjusted_indices

                # data_array_previous = np.empty(0, dtype=np.uint16)
                for num_line_adc in range(12):
                    if num_line_adc == 1 or num_line_adc == 8 or num_line_adc == 9 or num_line_adc == 5: continue
                    adc2read = num_line_adc + 1
                    detector = ret_adc_name(num_line_adc)
                    if num_line_adc <= 4 :
                        data = cube_one_pass_pixe[num_line_adc]
                    elif num_line_adc == 6 or num_line_adc == 7:
                        data = cube_one_pass_rbs[num_line_adc-6]
                    elif num_line_adc == 10 or num_line_adc == 11:
                        data = cube_one_pass_gamma[num_line_adc-10]

                    AGLAEFile.feed_hdf5_map(data, path_lst, detector, "FinalHDF", adc2read, sizeX, sizeY,nbcanaux,num_pass_y)
                print("\n")


def getSize(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size

def get_x_to_exclude(croissant,columns,last_x_value,first_x_value,change_line,fin_lst):
    """Recupére X pour exclure dans ce process"""
    # if columns== False:
    #     find_x = last_x_value
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

def get_colums_range(croissant,first_x_value,last_x_value):
    """True si plusieurs column dans le Data array"""
    if croissant== True:
        columns = last_x_value -1 > first_x_value
        # find_x = last_x_value
        # if columns==True: find_x -=1 # recherche X -1
    else:
        if first_x_value < last_x_value :
            columns = first_x_value + 1 < last_x_value
        else:
            columns = False
        # find_x = first_x_value
        
        # if columns==True: 
        #         first_x_value +=1
    return columns

def get_x_end_line_scan(croissant,sizex):
    """Get le X max du scan suivant ligne pair/impaire"""
    if croissant==True:
        end_x=sizex-1
    else:
        end_x=0
    return end_x

def look_if_end_lst(max_val_y_lue,sizeY,val_x_fin_map,val_fin_x):
    """informe si on atteins la fin du fichier LST"""
    if max_val_y_lue==sizeY-1 and val_x_fin_map == val_fin_x: #Fin du fichier LST ?
        fin_lst = True
    else: 
        fin_lst = False
    return fin_lst

def look_if_next_line(max_val_y_lue,y_scan_total):
    """informe si le dataset contient la fin du scan """
    
    if max_val_y_lue > y_scan_total: 
        change_line= True
    else:
        change_line =False

    return change_line
    # if last_x_value == 0:
    #             last_x_value = last_x_value
    
    # if croissant==True: # Donne dernière valeur X du Dataset 
    #     val_x_fin_map = last_x_value
    #     val_fin_x = sizeX -1
    # else:
    #     if max_val_y_lue > y_scan_total and first_x_value > last_x_value: # Contient X du prochaine scan
    #         val_x_fin_map = 0
    #         val_fin_x = 0    
    #     else:
    #         val_x_fin_map = first_x_value
    #         val_fin_x = 0
    
    

    # if max_val_y_lue > y_scan_total: # Changement de ligne de scan
    #     if val_x_fin_map == val_fin_x or (first_x_value > last_x_value and croissant==True) or (first_x_value > last_x_value and croissant==True): # Le dataset contient une partie du la ligne de retour -> 
    #         change_line= True
    # else:
    #     change_line= False

                            
def ret_num_adc(detector):
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

def ret_adc_name(num_adc):
   switcher = {
               0: "X1",
               1: "X2", # OFF
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
                            }
   return switcher.get(num_adc)

def ret_range_bytes(val):
    """Donne n°bits max pour valeur"""
    for bits in range(16):
        if val & (0b0000000000000001 << bits):
            nombre_bytes = bits
    return  2**(nombre_bytes+1) - 1

class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """

    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape)

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()



def main(args):
    print(args[0])

if __name__ == "__main__":
   lst_arg = sys.argv
   if len(lst_arg) >1:
        map_parameter = AGLAEFile.open_header_lst(lst_arg[1])
        AGLAEFile.extract_lst_vector(lst_arg[1],map_parameter,ADC_X = 8, ADC_Y = 9,)