import csv
from pymatgen.ext.matproj import MPRester
from pymatgen.entries.compatibility import MaterialsProjectCompatibility


# Replace with your Materials Project API key
API_KEY = 's'

# Input and output CSV file names
input_csv = 'TrainDataTest.csv'
output_csv = 'output2.csv'



# Function to get properties from the Materials Project API
def get_properties(formula):
    with MPRester(API_KEY) as mpr:
        try:
            entries = mpr.get_entries(formula)
            if entries:
                # Filter entries according to Materials Project compatibility
                compat = MaterialsProjectCompatibility()
                entries = compat.process_entries(entries)
                if entries:
                    entry = entries[0]
                    material_id = entry.entry_id
                    lattice_params = entry.structure.lattice.abc
                    space_group = entry.structure.get_space_group_info()[1]

                    # Query for elastic constants and surface properties
                    summary = mpr.summary.search(material_id, band_gap_min=0, band_gap_max=10)
                    properties = summary[0] if summary else None

                    if properties:
                        elastic_constants = properties.get('elasticity', None)
                        surface_properties = properties.get('surface_properties', None)
                    else:
                        elastic_constants = None
                        surface_properties = None

                    return lattice_params, space_group, elastic_constants, surface_properties
                else:
                    return None, None, None, None
            else:
                return None, None, None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None

# Read the CSV file containing molecule formulas
with open(input_csv, 'r') as input_file:
    reader = csv.reader(input_file)
    header = next(reader)
    header.extend(['a', 'b', 'c', 'space_group', 'elastic_constants', 'surface_properties'])

    # Write the output CSV file
    with open(output_csv, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)

        for row in reader:
            formula = row[0]  # Assuming the molecule formula is in the first column
            lattice_params, space_group, elastic_constants, surface_properties = get_properties(formula)

            if lattice_params and space_group:
                row.extend(lattice_params)
                row.append(space_group)
            else:
                row.extend([None, None, None, None])

            row.append(elastic_constants)
            row.append(surface_properties)

            writer.writerow(row)

print(f"Output written to {output_csv}")