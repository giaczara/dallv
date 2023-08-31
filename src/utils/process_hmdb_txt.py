from os import listdir, system, makedirs
from os.path import exists, join

input_folder = "../../txt/hmdb51_temp"
output_folder = "../../txt/hmdb51"
if exists(output_folder):
    system("rm -r {}".format(output_folder))
makedirs(output_folder)
classes = listdir("/data/hmdb51")

for split in [1, 2, 3]:
    train_split_file = open(join(output_folder, "train_split{}.txt".format(split)), "w")
    test_split_file = open(join(output_folder, "test_split{}.txt".format(split)), "w")
    for i, c in enumerate(sorted(classes)):
        class_file = open(join(input_folder, "{}_test_split{}.txt".format(c, split)), "r")
        for line in class_file:
            path, split_id = line.split()
            if int(split_id) == 1:
                train_split_file.write("{}/{} {}\n".format(c, path, i))
            elif int(split_id) == 2:
                test_split_file.write("{}/{} {}\n".format(c, path, i))
        class_file.close()
    train_split_file.close()
    test_split_file.close()
