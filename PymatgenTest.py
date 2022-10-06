import pymatgen.core

# Iterator test

PymeSet = ("electron_affinity",
           "molar_volume",
           "Z")

PymeIt = iter(PymeSet)

for PymeItLoop in PymeIt:
    pyme = pymatgen.core.periodic_table.Element
    print(pyme("Fe")PymeItLoop)
    continue


print(pyme("Fe").electron_affinity)

