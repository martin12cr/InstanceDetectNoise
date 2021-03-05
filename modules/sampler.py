import os

from experiment_utils.data_manager import subsample_file

base_path = "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/"
source_path = base_path + "Slow/"
target_path = base_path + "test_slow/"

for f in os.listdir(source_path):

    subsample_file(source_path + f, target_path + f.split('.')[0] + ".dat", t_file_size=0.5)


#subsample_file("/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/Slow/transcoding.dat", "/home/erick/google_drive/PARMA/InstanceNoiseDetection/GitRepo/Dataset/test.csv")