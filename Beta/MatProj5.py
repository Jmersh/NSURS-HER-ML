import csv
from pymatgen.ext.matproj import MPRester
from pymatgen.entries.compatibility import MaterialsProjectCompatibility


# Replace with your Materials Project API key
API_KEY = 's'

# Input and output CSV file names
input_csv = 'MatGenOutputTest.csv'
output_csv = 'MatGenOutputLattice.csv'



# Function to get lattice parameters and space group from the Materials Project API
def get_lattice_parameters_and_space_group(formula):
    with MPRester(API_KEY) as mpr:
        try:
            entries = mpr.get_entries(formula)
            if entries:
                # Filter entries according to Materials Project compatibility
                compat = MaterialsProjectCompatibility()
                entries = compat.process_entries(entries)
                if entries:
                    entry = entries[0]
                    lattice_params = entry.structure.lattice.abc
                    space_group = entry.structure.get_space_group_info()[1]
                    return lattice_params, space_group
                else:
                    return None, None
            else:
                return None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None

# Read the CSV file containing molecule formulas
with open(input_csv, 'r') as input_file:
    reader = csv.reader(input_file)
    header = next(reader)
    header.extend(['a', 'b', 'c', 'space_group'])

    # Write the output CSV file
    with open(output_csv, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)

        for row in reader:
            formula = row[0] # Assuming the molecule formula is in the first column
            lattice_params, space_group = get_lattice_parameters_and_space_group(formula)

            if lattice_params and space_group:
                row.extend(lattice_params)
                row.append(space_group)
            else:
                row.extend([None, None, None, None])

            writer.writerow(row)

print(f"Output written to {output_csv}")