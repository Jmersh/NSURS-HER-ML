import csv


def is_row_blank(row):
    return all(cell.strip() == '' for cell in row)


def has_missing_data(row):
    return any(cell.strip() == '' for cell in row)


def remove_blank_and_incomplete_rows(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header row and skip processing it
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if not is_row_blank(row) and not has_missing_data(row):
                writer.writerow(row)


if __name__ == '__main__':
    input_file = 'MatGenoutput.csv'
    output_file = 'MatGenOutputLatticeClean.csv'
    remove_blank_and_incomplete_rows(input_file, output_file)
    print(f"Cleaned CSV data has been written to {output_file}.")
