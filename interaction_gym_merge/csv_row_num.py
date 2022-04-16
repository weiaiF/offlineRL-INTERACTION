import csv

if __name__ == "__main__":
    total_row = 0
    for i in range(2):
        filename = 'collected_id_pair/DR_CHN_Roundabout_LN/DR_CHN_Roundabout_LN_' + str(i) + '_id_pair.csv'
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            total_row += len(list(csv_reader))

    print(total_row)

    print(895+388+8755+6242+908+8283+839+9)