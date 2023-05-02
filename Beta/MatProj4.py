import csv
from pymatgen import *
from pymatgen.core.structure import Structure
import pandas as pd

# Set up Materials Project API key
from pymatgen.ext.matproj import MPRester

api_key = "s"
mpr = MPRester(api_key)

# Load csv file into pandas dataframe
df = pd.read_csv('TrainDataTest.csv')

# Add a new column to the dataframe for material IDs
df['material_id'] = None

# Loop through each row of the dataframe and get material ID using materialsproject API
for index, row in df.iterrows():
    formula = row[0]
    data = mpr.query(criteria={"pretty_formula": formula}, properties=["material_id"])
    if data:
        material_id = data[0]['material_id']
        df.at[index, 'material_id'] = material_id

# Save updated dataframe to csv file
df.to_csv('TrainDataTest_updated.csv', index=False)
