import pandas as pd
from pymatgen import Molecule
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import DictSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher, \
    FrameworkComparator
from pymatgen.analysis.diffraction import xrd_from_pwdr, XRDCalculator

# Load the dataframe from a CSV file
df = pd.read_csv("molecules.csv")

# Define the Materials Project API key and instantiate an MPRester object
api_key = "your-api-key-here"
mpr = MPRester(api_key)

# Set up DFT calculation parameters
incar_params = {
    "ISIF": 2,  # Converge both ions and cell shape
    "EDIFFG": -0.01,  # Converge until forces are below this threshold
    "ENCUT": 520,  # Planewave cutoff energy (eV)
    "EDIFF": 1e-06,  # Electronic energy convergence criterion
    "ALGO": "Fast",  # Algorithm for electronic minimization
    "IBRION": 2,  # Conjugate gradient algorithm for ionic relaxation
    "NSW": 100,  # Maximum number of ionic steps
    "LREAL": False,  # Turn off real-space projection for better performance
    "LWAVE": False,  # Don't write WAVECAR file
    "LCHARG": False,  # Don't write CHGCAR file
}

# Loop through each molecule in the dataframe
for index, row in df.iterrows():
    # Create a Pymatgen Molecule object from the formula and lattice parameters
    molecule = Molecule.from_dict(row.to_dict())

    # Generate a VASP input set for the molecule using Materials Project relaxation parameters
    # Note: You may need to change the functional and POTCAR family depending on your specific use case
    vasp_input = MPRelaxSet(molecule, force_gamma=True, user_incar_settings=incar_params,
                            functional="PBE", potcar_functional="PBE", potcar_family="pbe")

    # Write the VASP input files to disk
    vasp_input.write_input(".")

    # Run the VASP calculation
    !mpirun - n
    4
    vasp_std > vasp.out

    # Parse the VASP output using Pymatgen's Vasprun class
    vasprun = Vasprun("vasprun.xml")

    # Get the final structure and space group of the relaxed molecule
    final_structure = vasprun.final_structure
    sga = SpacegroupAnalyzer(final_structure)
    space_group = sga.get_space_group_symbol()

    # Calculate the XRD pattern of the relaxed molecule
    xrd_calculator = XRDCalculator()
    xrd = xrd_calculator.get_pattern(final_structure)

    # Add the space group and XRD data to the dataframe
    df.at[index, "space_group"] = space_group
    df.at[index, "xrd"] = str(xrd))

# Save the modified dataframe to a new CSV file