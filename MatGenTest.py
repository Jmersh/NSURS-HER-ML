import periodictable as pt
from periodictable import *
import pandas as pd
import csv
#from mp_api.client import MPRester
import pymatgen


# Materials List Composition

M_MatN = 0
X_MatN = 0
M_Mat = {"Ti",
         "Hf",
         "V",
         "Nb",
         "Ta",
         "Cr",
         "Mo",
         "W",
         "Mn",
         "Tc",
         "Sc",
         "Zr",
         "Ru",
         "Fe",
         "Ni",
         "Rh",
         "Os",
         "Co",
         "Ir",
         "Re"}

X_Mat = {"B",
         "N",
         "C"}

target = {39: None, 91: None, 93: None}
M_Mat_C = (str(M_Mat).translate(target))
X_Mat_C = (str(X_Mat).translate(target))
# Testing mp-api

# with MPRester("your_api_key_here") as mpr:

# CSV Writer

with open('MatGenTest.csv', mode='w') as MatGenTestCSV:
    MatGenTestCSVWrite = csv.writer(MatGenTestCSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    MatGenTestCSVWrite.writerow(["Formula", "Metal Symbol", "M Number", "X Symbol", "X Number", "Metal Mass", "Metal Density", "Metal Interatomic Distance", "Metal Covalent Radius", "X Mass","X Density", "X Interatomic Distance", "X Covalent Radius"])
    for M_MatN2 in M_Mat:
        for X_MatN2 in X_Mat:
            ps = pt.elements.symbol
            MatGenTestCSVWrite.writerows([((M_MatN2 + "1" + X_MatN2 + "1"), M_MatN2, "1", *X_MatN2, "1", ps(M_MatN2).mass, ps(M_MatN2).density, ps(M_MatN2).interatomic_distance, ps(M_MatN2).covalent_radius, ps(X_MatN2).mass, ps(X_MatN2).density, ps(X_MatN2).interatomic_distance, ps(X_MatN2).covalent_radius)])
            MatGenTestCSVWrite.writerows([((M_MatN2 + "1" + X_MatN2 + "2"), M_MatN2, "1", *X_MatN2, "2", ps(M_MatN2).mass, ps(M_MatN2).density, ps(M_MatN2).interatomic_distance, ps(M_MatN2).covalent_radius, ps(X_MatN2).mass, ps(X_MatN2).density, ps(X_MatN2).interatomic_distance, ps(X_MatN2).covalent_radius)])
            MatGenTestCSVWrite.writerows([((M_MatN2 + "2" + X_MatN2 + "1"), M_MatN2, "2", *X_MatN2, "1", ps(M_MatN2).mass, ps(M_MatN2).density, ps(M_MatN2).interatomic_distance, ps(M_MatN2).covalent_radius, ps(X_MatN2).mass, ps(X_MatN2).density, ps(X_MatN2).interatomic_distance, ps(X_MatN2).covalent_radius)])
            MatGenTestCSVWrite.writerows([((M_MatN2 + "3" + X_MatN2 + "4"), M_MatN2, "3", *X_MatN2, "4", ps(M_MatN2).mass, ps(M_MatN2).density, ps(M_MatN2).interatomic_distance, ps(M_MatN2).covalent_radius, ps(X_MatN2).mass, ps(X_MatN2).density, ps(X_MatN2).interatomic_distance, ps(X_MatN2).covalent_radius)])
            MatGenTestCSVWrite.writerows([((M_MatN2 + "5" + X_MatN2 + "2"), M_MatN2, "5", *X_MatN2, "2", ps(M_MatN2).mass, ps(M_MatN2).density, ps(M_MatN2).interatomic_distance, ps(M_MatN2).covalent_radius, ps(X_MatN2).mass, ps(X_MatN2).density, ps(X_MatN2).interatomic_distance, ps(X_MatN2).covalent_radius)])
            continue
