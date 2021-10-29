import csv  
import random
import config

header = ['sentence_a', 'sentence_b']
data = [[1,2,3,4], [5,6,7,8]]
max_length = config.max_length
entry_num = 10000

with open('numbers.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    # writer.writerow(data)

    for _ in range(entry_num):
        s = random.randint(1, max_length/2)
        len = random.randint(1, max_length/4)
        data[0] = [i for i in range(s, s+len)]
        data[1] = [i for i in range(s+len, s+2*len)]
        writer.writerow(data)


        
