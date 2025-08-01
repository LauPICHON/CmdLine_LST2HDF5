Fonctions: cmdLine_lst_vector.exe
Converti le fichier LST créée par le multicanal MPAWIN3 pour recréer les cartographies au format hdf5.
Extrait les fichiers ascii pour les convertir au format hdf5.

Usage:  cmdLine_lst_vector.exe  <arg1:Path of LST file> <arg2: type of extraction 'maps' or 'spectra'> <arg3:path of one ASCII spectra>")
Exemple: 
cmdLine_lst_vector.exe maps "C:\\Data\\2025\\test_IBIL\\20250630_0029_OBJ_SRV-1_IBA.lst" “C:\\Data\\2025\\test_IBIL\\20250630_SRV-Vishnu\\20250630_0001_STD_SRV-1_IBA.x0"

Le fichier "config_lst2hdf5.ini" permet: 
- de définir le nombre de canaux pour chaque type d'analyse (PIXE,RBS,GAMMA)
- De définir sur quel voie du multicanal les coordonnées X & Y sont enregistrés.


Fonctions :

1)	Hdf5 des analyses ponctuelles, arg2=spectra (standard et batch)

Deux fichiers hdf5 sont créés automatiquement à la fin des analyses correspondant respectivement, aux analyses définies comme un standard (standard) et celles sur les objets/échantillons (batch).
Le programme python va lire les fichiers ascii (x0,x1,x2,…rbs150,g70,…) pour créer les dataset de l’hdf5 et extraire les metadata depuis le fichier lst.
-	Nommage : « date_nomproject_standard_IBA.hdf5 »
ex : 20250630_SRV-1_standard_IBA.hdf5

-	Nommage : « date_nomproject_batch_IBA.hdf5 »
ex : 20250630_SRV-1_batch_IBA.hdf5

2)	Hdf5 des cartographies, arg2=maps (standard et batch)

Pour chaque carto un fichier hdf5 est créé automatiquement à partir du fichier lst (mpawin) sur le PC d’acquisition (MPA4-ACQ) puis copier dans le dossier « HDF5_maps_files » du NAS3-AGLAE
-	Nommage : « nom du fichier lst.hdf5 »
ex : 20250630_0029_OBJ_SRV-1_IBA.hdf5

Cas particulier IBIL.
Pour le moment le programme labview IBIL enregistre les cartos dans le format EDF. Lors de la création du fichier hdf5, mon programme python va rechercher le dossier IBIL (dans le dossier de l’utilisateur) 
contenant les fichiers EDF pour les lires et ainsi ajouter le dataset IBIL à l’hdf5 des autres cartos.
