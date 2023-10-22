import csv

date = '01-10-2023'
data = []

with open('output.csv') as f:
    reader = csv.reader(f, delimiter=',')
    skip_first_row = True
    for row in reader:
        if skip_first_row:
            skip_first_row = False
            continue
        name, max_frequency, max_magnitude = row[0], row[1], row[2]
        name = int(name.split('.')[0])
        data.append([name, max_frequency, max_magnitude])

data.sort()
print(data)

header = ['No.', 'Max Frequency (Hz)',	'Max Magnitude']

with open(f'max_frequency_{date}.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data)

