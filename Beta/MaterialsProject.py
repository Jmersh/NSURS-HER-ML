import pandas as pd
from pymatgen.ext.matproj import MPRester

# Set up Materials Project API key and instantiate MPRester object
api_key = "s"
mpr = MPRester(api_key)


# Function to retrieve Materials Project ID for a given formula
def get_mp_id(formula):
    # Search Materials Project for materials with matching formula
    results = mpr.summary.search(formula)
    # Extract Material ID from first result
    if len(results) > 0:
        return results[0]["material_id"]
    else:
        return None


# Example usage of get_mp_id function
print(get_mp_id("Fe2O3"))  # Should print "mp-1347"

# Example Pandas dataframe with molecular formulas
df = pd.DataFrame({"Formula": ["NaCl", "Fe2O3", "CuSO4"]})

# Add Material ID column to dataframe
df["Material ID"] = df["Formula"].apply(get_mp_id)

# Print resulting dataframe
print(df)

# Define list of Materials Project IDs for desired materials
mp_ids = ["mp-1234", "mp-5678", "mp-9101"]

# Initialize empty list to store lattice parameters
lattice_params = []

# Loop through Materials Project IDs and retrieve lattice parameters
for mp_id in mp_ids:
    # Retrieve structure object for material
    structure = mpr.get_structure_by_material_id(mp_id)
    # Extract lattice parameters from structure object
    lattice_params.append(structure.lattice.parameters)

# Convert list of lattice parameters to Pandas dataframe
df = pd.DataFrame(lattice_params, columns=["a", "b", "c", "alpha", "beta", "gamma"])

# Print the resulting dataframe
print(df)
