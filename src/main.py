# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import tensorflow
import csv
import cloud
import codecs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    maxInt = 922337203
    csv.field_size_limit(maxInt)

    fake_output = open("fake-list.txt","w",encoding='utf-8', errors='ignore')


    with codecs.open("datasets/gossipcop_fake.csv", "r", encoding='utf-8', errors='ignore') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fake_output.write(row[2])
    file.close()

    with codecs.open("datasets/politifact_fake.csv", "r", encoding='utf-8', errors='ignore') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fake_output.write(row[2])
    file.close()
    fake_output.close()
# -----------------------------------------------------------------------------------------------
    real_output = open("real-list.txt", "w", encoding='utf-8', errors='ignore')

    with codecs.open("datasets/gossipcop_real.csv", "r", encoding='utf-8', errors='ignore') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            real_output.write(row[2])
    file.close()

    with codecs.open("datasets/politifact_real.csv", "r", encoding='utf-8', errors='ignore') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            real_output.write(row[2])
    file.close()
    real_output.close()

    cloud.make_cloud("real-list.txt")
    cloud.make_cloud("fake-list.txt")

