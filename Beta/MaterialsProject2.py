import pandas as pd
from pymatgen.ext.matproj import MPRester

# Set up Materials Project API key and instantiate MPRester object
api_key = "s"
mpr = MPRester(api_key)

# Define list of elements to search for
elements = ["Li", "Fe", "O"]

# Search Materials Project for structures containing specified elements
results = mpr.summary.search(elements=elements)

# Extract lattice parameters from results
lattices = []
for r in results:
    lattices.append({
        "material_id": r["material_id"],
        "formula": r["pretty_formula"],
        "a": r["structure"]["lattice"].a,
        "b": r["structure"]["lattice"].b,
        "c": r["structure"]["lattice"].c,
        "alpha": r["structure"]["lattice"].alpha,
        "beta": r["structure"]["lattice"].beta,
        "gamma": r["structure"]["lattice"].gamma
    })

# Create Pandas dataframe from list of lattices
df = pd.DataFrame(lattices)

# Print resulting dataframe
print(df)
