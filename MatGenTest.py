import periodictable as pt
from periodictable import *
import pandas as pd
import csv

# Materials List Composition

M_MatN = 0
X_MatN = 0
M_Mat = ([["Ti"],
          ["Hf"],
          ["V"],
          ["Nb"],
          ["Ta"],
          ["Cr"],
          ["Mo"],
          ["W"],
          ["Mn"],
          ["Tc"],
          ["Sc"],
          ["Zr"],
          ["Ru"],
          ["Fe"],
          ["Ni"],
          ["Rh"],
          ["Os"],
          ["Co"],
          ["Ir"],
          ["Re"]])

X_Mat = ([["B"],
          ["N"],
          ["C"]])

target = {39: None, 91: None, 93: None}
M_Mat_C = (str(M_Mat).translate(target))
X_Mat_C = (str(X_Mat).translate(target))

# Material Print (Simple test of the loop for generating materials)

for M_MatN2 in M_Mat:
    for X_MatN2 in X_Mat:
        print(*M_MatN2, *X_MatN2, "2", sep='')
        continue
for M_MatN2 in M_Mat:
    for X_MatN2 in X_Mat:
        print(*M_MatN2, "3", *X_MatN2, "4", sep='')
        continue
for M_MatN2 in M_Mat:
    for X_MatN2 in X_Mat:
        print(*M_MatN2, *X_MatN2, sep='')
        continue
for M_MatN2 in M_Mat:
    for X_MatN2 in X_Mat:
        print(*M_MatN2, "2", *X_MatN2, sep='')
        continue
for M_MatN2 in M_Mat:
    for X_MatN2 in X_Mat:
        print(*M_MatN2, "5", *X_MatN2, "2", sep='')
        continue

# Periodic Table Library (Insert

B = pt.B
N = pt.N
C = pt.C

# CSV Writer

with open('MatGenTest.csv', mode='w') as MatGenTestCSV:
    MatGenTestCSVWrite = csv.writer(MatGenTestCSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for M_MatN2 in M_Mat:
        for X_MatN2 in X_Mat:
            MatGenTestCSVWrite.writerows([(*M_MatN2, "1", *X_MatN2, "1")])
            MatGenTestCSVWrite.writerows([(*M_MatN2, "1", *X_MatN2, "2")])
            MatGenTestCSVWrite.writerows([(*M_MatN2, "2", *X_MatN2, "1")])
            MatGenTestCSVWrite.writerows([(*M_MatN2, "3", *X_MatN2, "4")])
            MatGenTestCSVWrite.writerows([(*M_MatN2, "5", *X_MatN2, "2")])
            continue
