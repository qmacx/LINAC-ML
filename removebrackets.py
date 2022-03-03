#Removing brackets from csv

import numpy as np
import os
from csv import writer
from csv import reader

#Open file
with open('data1.csv') as f, open('sampleCx.txt', 'w') as f2:
    for line in f:
        line = line.strip()
        if not line or len(line) >= 170:
            f2.write(line + '\n')
            
with open('sampleCx.txt','r') as p:
    file = p.readlines()

#Choose columns with brackets - append values into new list
angle1= []
o7x = []
for l in file:
    o7x.append(l.split(',')[15])
    angle1.append(l.split(',')[24])

#Remove brackets
o7x = [item.replace("]", "") for item in o7x]
o7x = [item.replace('\n', "") for item in o7x]
angle1= [item.replace("[", "") for item in angle1]


#Delete old columns with brackets (work backwards)
with open("data1.csv","rt") as source:
    rdr= csv.reader( source )
    with open("data2.csv","w") as result:
        wtr= csv.writer( result )
        for r in rdr:
            del r[24]
            del r[15]
            wtr.writerow( r )


def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)


#Add lists to the end of the csv file
add_column_in_csv('data2.csv', 'data3.csv', lambda row, line_num: 
row.append(angle1[line_num - 1]))
