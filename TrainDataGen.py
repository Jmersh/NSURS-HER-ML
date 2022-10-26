import periodictable as pt
# from periodictable import *
import csv
# import pymatgen
import pymatgen.core
import pandas as pd

testcsv = "TrainingData.csv"
testpdarray = pd.read_csv(testcsv, usecols=(1, 2))

with open('TrainDataOutput.csv', mode='w', newline='') as TrainDataCsv:
    MatGenTestCSVWrite = csv.writer(TrainDataCsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Create headers in CSV file
    MatGenTestCSVWrite.writerow([
        "", "Metal Symbol", "M Number", "X Symbol", "X Number",
        "Metal Sym Number", "Metal Group", "Metal Row",
        "Metal Mass", "Metal Density", "Metal Inter-atomic Distance",
        "Metal Covalent Radius", "Metal First Ionization Energy",
        "Metal Electron Affinity",
        "X Sym Number", "X Group", "X Row",
        "X Mass", "X Density", "X Inter-atomic Distance",
        "X Covalent Radius", "X First Ionization Energy",
        "X Electron Affinity",
    ])

    for i in range(len(testpdarray.index)):  # Loops through the range of the arrays index
        for idx, row in testpdarray.loc[i:i].iterrows():
            M_MatT = testpdarray.loc[idx].iat[0]
            X_MatT = testpdarray.loc[idx].iat[1]
            ps = pt.elements.symbol  # Declare ps as shorthand for longer function
            pyme = pymatgen.core.periodic_table.Element  # Pymatgen implementation
            MatGenTestCSVWrite.writerows([((M_MatT + X_MatT), M_MatT, "1", X_MatT, "1",
                                           pyme(M_MatT).number, pyme(M_MatT).group, pyme(M_MatT).row,
                                           ps(M_MatT).mass, ps(M_MatT).density, ps(M_MatT).interatomic_distance,
                                           ps(M_MatT).covalent_radius, pyme(M_MatT).ionization_energy,
                                           pyme(M_MatT).electron_affinity,
                                           pyme(M_MatT).number, pyme(X_MatT).group, pyme(X_MatT).row,
                                           ps(X_MatT).mass, ps(X_MatT).density, ps(X_MatT).interatomic_distance,
                                           ps(X_MatT).covalent_radius, pyme(X_MatT).ionization_energy,
                                           pyme(X_MatT).electron_affinity)])
        continue
    # Please note that the first column will be wrongly generated along with the M and X number ratios, these we hand
    # edited.
    print("CSV Output complete")
