from pytube import YouTube
from tqdm import tqdm

train_file_path = "../../data/sports1m_msda_train.txt"
test_file_path = "../../data/sports1m_msda_test.txt"
train_file = open(train_file_path, "r")
test_file = open(test_file_path, "r")
error_log = open("error_log.txt", "w")

# for split, txt_file_path, txt_file in zip(
#     ["train", "test"], [train_file_path, test_file_path], [train_file, test_file]
# ):
for split, txt_file_path, txt_file in zip(
        ["train"], [train_file_path], [train_file]
):
    file_length = len([n for n in txt_file])
    txt_file.close()
    print("Downloading {}".format(split))
    txt_file = open(txt_file_path, "r")
    for line in tqdm(txt_file, total=file_length):
        split_line = line.split()
        file_path = split_line[-1]
        youtube_id = file_path.split("/")[1].split(".")[0]
        youtube_link = "https://www.youtube.com/watch?v={}".format(youtube_id)
        yt = YouTube(youtube_link)
        try:
            ys = yt.streams.get_highest_resolution()
            ys.download(
                "/data/sports_da/sports1m/{}/{}.mp4".format(split, youtube_id)
            )
        except Exception as e:
            print("Error downloading {} with error {}".format(youtube_id, e))
            error_log.write("Error downloading {} with error {}".format(youtube_id, e))
            continue

    txt_file.close()

error_log.close()
