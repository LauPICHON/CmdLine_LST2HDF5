
**Functions:** `cmdLine_lst_vector.exe`
Converts the LST file created by the MPAWIN3 multichannel system to recreate the maps in hdf5 format. 
It extracts ASCII files and converts them to hdf5 format.

**Usage:**
`cmdLine_lst_vector.exe <arg1:Path of LST file> <arg2: type of extraction 'maps' or 'spectra'> <arg3:path of one ASCII spectra>")

**Example:**
`cmdLine_lst_vector.exe maps "C:\Data\2025\test_IBIL\20250630_0029_OBJ_SRV-1_IBA.lst" “C:\Data\2025\test_IBIL\20250630_SRV-Vishnu\20250630_0001_STD_SRV-1_IBA.x0"`

The file `config_lst2hdf5.ini` allows you to:
- Define the number of channels for each type of analysis (PIXE, RBS, GAMMA).
- Specify on which multichannel channel the X & Y coordinates are recorded. (COORD_X & COORD_Y values)

**Functions:**

- **Hdf5 for spot analyses, arg2=spectra (standard and batch):**
    - Two hdf5 files are automatically created at the end of the analyses, corresponding respectively to analyses defined as a standard ("standard") and those on objects/samples ("batch").
    - The Python program will read the ASCII files (x0, x1, x2, … rbs150, g70, …) to create the hdf5 datasets and extract metadata from the lst file.
    - Naming convention: `"date_projectname_standard_IBA.hdf5"` e.g., `20250630_SRV-1_standard_IBA.hdf5`
    - Naming convention: `"date_projectname_batch_IBA.hdf5"` e.g., `20250630_SRV-1_batch_IBA.hdf5`

- **Hdf5 for imaging/mappings, arg2=maps (standard and batch):**
    - For each map, one hdf5 file is automatically created from the lst (mpawin) file on the acquisition PC (MPA4-ACQ) then copied to the "HDF5_maps_files" folder on NAS3-AGLAE.
    - Naming convention: `"name of the lst file.hdf5"` e.g., `20250630_0029_OBJ_SRV-1_IBA.hdf5`

- **Special case: IBIL.**
    - For now, the IBIL Labview program records the maps in EDF format.
    - When creating the hdf5 file, my Python program will search for the IBIL folder (within the user's folder) containing the EDF files in order to read them and add the IBIL dataset to the hdf5 file along with the other maps.
