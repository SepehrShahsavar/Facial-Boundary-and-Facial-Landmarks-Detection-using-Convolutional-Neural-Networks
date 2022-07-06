import cv2 as cv
import os

ann = open("./Data/annotations.txt", 'r')

new_ann = open("./newer_ann.txt", "w")

data_path = "./Data/train/"

lines = ann.readlines()

pointer = 0
txt = ""

counter = 0
while pointer != len(lines):
    # direct = lines[pointer][:-1].split("/")
    # old_path = data_path + lines[pointer][:-1]
    # new_path = data_path + direct[0] + '/' + str(counter) + '.jpg'
    # print(old_path)
    # print(new_path)
    # os.rename(old_path,new_path)
    # txt += new_path + "\n"
    pointer += 1
    n = int(lines[pointer])
    txt += lines[pointer]
    pointer += 1
#
    for i in range(n):
        txt += lines[pointer]
        pointer += 1
#
    counter += 1
#
new_ann.write(txt)

new_ann.close()