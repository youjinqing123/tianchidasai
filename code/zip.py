import zipfile

f1 = zipfile.ZipFile("../data/First_round_data/jinnan2_round1_train_20190305.zip",'r')
for file in f1.namelist():
    f1.extract(file,"../data/First_round_data/")


f3 = zipfile.ZipFile("../data/First_round_data/jinnan2_round1_test_b_20190326.zip",'r')
for file in f3.namelist():
    f3.extract(file,"../data/First_round_data/")



