from os.path import join, exists
from os import makedirs, system

class_names = {
    "climb": 0,
    "fencing": 1,
    "golf": 2,
    "kick_ball": 3,
    "pullup": 4,
    "punch": 5,
    "pushup": 6,
    "ride_bike": 7,
    "ride_horse": 8,
    "shoot_ball": 9,
    "shoot_bow": 10,
    "walk": 11,
}

annotations_path = "/data/image_to_video/hmdb51/annotations/"
prefix = "/data/hmdb_ucf_source_free/hmdb/frames/"

txt_dir = "../../txt/hmdb_ucf_im2vid/"
if exists(txt_dir):
    system("rm -rf {}".format(txt_dir))
makedirs(txt_dir)


for split in ["1", "2", "3"]:
    main_train_file_path = join(txt_dir, "hmdb_train_split{}.txt".format(split))
    main_test_file_path = join(txt_dir, "hmdb_test_split{}.txt".format(split))
    count_train = -1
    count_test = -1
    for c in class_names:
        file = join(annotations_path, "{}_test_split{}.txt".format(c, split))
        file_stream = open(file, "r")
        for line in file_stream:
            line = line.strip()
            if line == "":
                continue
            line = line.split(" ")
            video_name = line[0]
            s = line[1]
            if s == "1":
                count_train += 1
                main_file = open(main_train_file_path, "a")
            elif s == "2":
                count_test += 1
                main_file = open(main_test_file_path, "a")
            else:
                continue
            main_file.write(
                "{}\t{}\t{}\n".format(
                    count_train if s == "1" else count_test,
                    class_names[c],
                    join(prefix, c, video_name[:-4]),
                )
            )
            main_file.close()

annotations_path = "/data/image_to_video/ucf101/annotations/"
prefix = "/data/hmdb_ucf_source_free/ucf/frames/"

class_names = {
    "RockClimbingIndoor": 0,
    "RopeClimbing": 0,
    "Fencing": 1,
    "GolfSwing": 2,
    "SoccerPenalty": 3,
    "PullUps": 4,
    "Punch": 5,
    "BoxingPunchingBag": 5,
    "BoxingSpeedBag": 5,
    "PushUps": 6,
    "Biking": 7,
    "HorseRiding": 8,
    "Basketball": 9,
    "Archery": 10,
    "WalkingWithDog": 11,
}

for split in ["1", "2", "3"]:
    for s in ["train", "test"]:
        main_file_path = join(txt_dir, "ucf_{}_split{}.txt".format(s, split))
        main_file = open(main_file_path, "w")
        file = join(annotations_path, "{}list0{}.txt".format(s, split))
        file_stream = open(file, "r")
        count = 0
        for line in file_stream:
            line = line.strip()
            if line == "":
                continue
            line = line.split(" ")
            video_name = line[0]
            c = video_name.split("/")[0]
            if c in class_names:
                main_file.write(
                    "{}\t{}\t{}\n".format(
                        count,
                        class_names[c],
                        join(prefix, video_name[:-4]),
                    )
                )
                count += 1
        main_file.close()

