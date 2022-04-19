# This code is adapted from https://github.com/darraghmaguire/automatic-scoliosis-assessment/blob/master/dataset-preprocessing/fixLandmarkErrors.py


import scipy.io
import os
import numpy as np

# Input data directories
train_label_dir = "./boostnet_labeldata/labels/training"

# label 4

label4 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-03-Jan-2017-163 B AP.jpg.mat"))["p2"]

temp1 = label4[60, 0]
temp2 = label4[60, 1]

label4[60, 0] = label4[62, 0]
label4[60, 1] = label4[62, 1]

label4[62, 0] = temp1
label4[62, 1] = temp2

temp1 = label4[61, 0]
temp2 = label4[61, 1]

label4[61, 0] = label4[63, 0]
label4[61, 1] = label4[63, 1]

label4[63, 0] = temp1
label4[63, 1] = temp2

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-03-Jan-2017-163 B AP.jpg.mat"), {"p2": label4})

# label 11

label11 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-05-Jan-2017-169 A AP.jpg.mat"))["p2"]

temp1 = label11[64, 0]
temp2 = label11[64, 1]

label11[64, 0] = label11[66, 0]
label11[64, 1] = label11[66, 1]

label11[66, 0] = temp1
label11[66, 1] = temp2

temp1 = label11[65, 0]
temp2 = label11[65, 1]

label11[65, 0] = label11[67, 0]
label11[65, 1] = label11[67, 1]

label11[67, 0] = temp1
label11[67, 1] = temp2

label11[:, :] = np.roll(label11[:, :], 4, axis=0)

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-05-Jan-2017-169 A AP.jpg.mat"), {"p2": label11})

# label 15
label15 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-05-Jan-2017-172 B AP.jpg.mat"))["p2"]

for k in range(22, 66):
    label15[k, 0] = label15[k + 2, 0]
    label15[k, 1] = label15[k + 2, 1]

label15[66, 0] = label15[64, 0] - 5
label15[66, 1] = label15[64, 1] + 10

label15[67, 0] = label15[65, 0] - 5
label15[67, 1] = label15[65, 1] + 10

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-05-Jan-2017-172 B AP.jpg.mat"), {"p2": label15})

# label 29
label29 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-06-Jan-2017-182 A AP.jpg.mat"))["p2"]

label29[:, :] = np.roll(label29[:, :], 2, axis=0)

for k in range(64):
    label29[67 - k, 0] = label29[67 - k - 2, 0]
    label29[67 - k, 1] = label29[67 - k - 2, 1]

for k in range(12):
    label29[67 - k, 0] = label29[67 - k - 2, 0]
    label29[67 - k, 1] = label29[67 - k - 2, 1]

label29[2, 1] = label29[2, 1] - 2
label29[3, 1] = label29[3, 1] - 2
label29[50, 1] = label29[50, 1] - 1
label29[51, 1] = label29[51, 1] - 2
label29[52, 1] = label29[52, 1] - 5
label29[53, 1] = label29[53, 1] - 5
label29[55, 1] = label29[55, 1] - 3
label29[56, 1] = label29[56, 1] + 3

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-06-Jan-2017-182 A AP.jpg.mat"), {"p2": label29})

# label 32
label32 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-06-Jan-2017-184 A AP.jpg.mat"))["p2"]

for k in range(18, 66):
    label32[k, 0] = label32[k + 2, 0]
    label32[k, 1] = label32[k + 2, 1]

label32[20, 1] = label32[20, 1] - 2
label32[21, 1] = label32[21, 1] - 2

label32[66, 0] = label32[64, 0] - 2
label32[66, 1] = label32[64, 1] + 10

label32[67, 0] = label32[65, 0] - 2
label32[67, 1] = label32[65, 1] + 10

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-06-Jan-2017-184 A AP.jpg.mat"), {"p2": label32})

# label 34
label34 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-06-Jan-2017-188 B AP.jpg.mat"))["p2"]

for k in range(42):
    label34[67 - k, 0] = label34[67 - k - 2, 0]
    label34[67 - k, 1] = label34[67 - k - 2, 1]

label34[24, 0] = label34[22, 0] + 1
label34[24, 1] = label34[22, 1] + 4
label34[25, 0] = label34[23, 0]
label34[25, 1] = label34[23, 1] + 3

label34[58, 1] = label34[58, 1] - 1
label34[59, 1] = label34[59, 1] - 1

for k in range(6):
    label34[67 - k, 0] = label34[67 - k - 2, 0]
    label34[67 - k, 1] = label34[67 - k - 2, 1]

label34[60, 1] = label34[60, 1] - 7
label34[61, 1] = label34[61, 1] - 7

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-06-Jan-2017-184 A AP.jpg.mat"), {"p2": label34})

# label 41
label41 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-194 A AP.jpg.mat"))["p2"]

label41[24, 1] = label41[24, 1] - 2
label41[25, 1] = label41[25, 1] - 2

label41[28, 1] = label41[28, 1] - 2
label41[29, 1] = label41[29, 1] - 2

label41[32, 1] = label41[32, 1] - 1
label41[33, 1] = label41[33, 1] - 1

label41[36, 1] = label41[36, 1] - 4
label41[37, 1] = label41[37, 1] - 4

label41[40, 1] = label41[40, 1] - 4
label41[41, 1] = label41[41, 1] - 4

label41[44, 1] = label41[44, 1] - 4
label41[45, 1] = label41[45, 1] - 4

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-194 A AP.jpg.mat"), {"p2": label41})

# label 73
label73 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-207 B AP.jpg.mat"))["p2"]

for k in range(64):
    label73[k, 0] = label73[k + 4, 0]
    label73[k, 1] = label73[k + 4, 1]

label73[64, 0] = label73[60, 0] - 5
label73[64, 1] = label73[60, 1] + 20

label73[65, 0] = label73[61, 0] - 5
label73[65, 1] = label73[61, 1] + 20

label73[66, 0] = label73[62, 0] - 5
label73[66, 1] = label73[62, 1] + 20

label73[67, 0] = label73[63, 0] - 5
label73[67, 1] = label73[63, 1] + 18

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-207 B AP.jpg.mat"), {"p2": label73})

# label 85
label85 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-215 D AP.jpg.mat"))["p2"]

for k in range(36):
    label85[67 - k, 0] = label85[67 - k - 2, 0]
    label85[67 - k, 1] = label85[67 - k - 2, 1]

label85[28, 1] = label85[26, 1] + 3
label85[29, 1] = label85[27, 1] + 3

label85[32, 1] = label85[32, 1] + 3
label85[33, 1] = label85[33, 1] + 3

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-215 D AP.jpg.mat"), {"p2": label85})

# label 92
label92 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-219 A AP.jpg.mat"))["p2"]

label92[4, 1] = label92[4, 1] - 2
label92[5, 1] = label92[5, 1] - 2

label92[8, 1] = label92[8, 1] - 2
label92[9, 1] = label92[9, 1] - 2

label92[12, 1] = label92[12, 1] - 3
label92[13, 1] = label92[13, 1] - 3

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-219 A AP.jpg.mat"), {"p2": label92})

# label 144
label144 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-243 B AP.jpg.mat"))["p2"]

for k in range(42, 56):
    label144[k, 0] = label144[k + 2, 0]
    label144[k, 1] = label144[k + 2, 1]

label144[44, 1] = label144[44, 1] - 1
label144[45, 1] = label144[45, 1] - 2

label144[56, 1] = label144[56, 1] + 6
label144[57, 1] = label144[57, 1] + 7

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-243 B AP.jpg.mat"), {"p2": label144})

# label 194
label194 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-12-Jan-2017-262 A AP.jpg.mat"))["p2"]

label194[0, 1] = label194[0, 1] - 10

label194[64, 0] = label194[60, 0] - 3
label194[64, 1] = label194[60, 1] + 20

label194[65, 0] = label194[61, 0]
label194[65, 1] = label194[61, 1] + 20

label194[66, 0] = label194[62, 0]
label194[66, 1] = label194[62, 1] + 20

label194[67, 0] = label194[63, 0]
label194[67, 1] = label194[63, 1] + 20

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-12-Jan-2017-262 A AP.jpg.mat"), {"p2": label194})

# label 219
label219 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-20-Jan-2017-275 D AP.jpg.mat"))["p2"]

for k in range(64):
    label219[67 - k, 0] = label219[67 - k - 4, 0]
    label219[67 - k, 1] = label219[67 - k - 4, 1]

label219[0, 0] = label219[0, 0]
label219[0, 1] = label219[0, 1] - 10

label219[1, 0] = label219[1, 0]
label219[1, 1] = label219[1, 1] - 11

label219[2, 0] = label219[2, 0]
label219[2, 1] = label219[2, 1] - 10

label219[3, 0] = label219[3, 0]
label219[3, 1] = label219[3, 1] - 10

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-20-Jan-2017-275 D AP.jpg.mat"), {"p2": label219})

# label 243
label219 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-3 A AP.jpg.mat"))["p2"]

label219[26, 0] = label219[26, 0] - 6
label219[28, 0] = label219[28, 0] - 6

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-20-Jan-2017-275 D AP.jpg.mat"), {"p2": label219})

# label 262
label262 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-coronal2.jpg.mat"))["p2"]

label262[22, 0] = label262[24, 0]
label262[22, 1] = label262[24, 1] - 3

label262[23, 0] = label262[25, 0]
label262[23, 1] = label262[25, 1] - 3

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-coronal2.jpg.mat"), {"p2": label262})

# label 266
label266 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-coronal6.jpg.mat"))["p2"]

label266[63, 0] = label266[65, 0]
label266[63, 1] = label266[65, 1] - 4

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-coronal6.jpg.mat"), {"p2": label266})

# label 269
label269 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-coronal9.jpg.mat"))["p2"]

label269[14, 0] = label269[16, 0] + 2
label269[14, 1] = label269[16, 1] - 3

label269[15, 0] = label269[17, 0]
label269[15, 1] = label269[17, 1] - 3

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-20-Jul-2016-coronal9.jpg.mat"), {"p2": label269})

# label 270
label270 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-21-Jul-2016-15 E AP.jpg.mat"))["p2"]

label270[54, 0] = label270[54, 0] + 5
label270[54, 1] = label270[54, 1] - 5

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-21-Jul-2016-15 E AP.jpg.mat"), {"p2": label270})

# label 314
label314 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-21-Nov-2016-140 B AP.jpg.mat"))["p2"]

label314[54, 1] = label314[54, 1] + 10
label314[55, 1] = label314[55, 1] + 10

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-21-Nov-2016-140 B AP.jpg.mat"), {"p2": label314})

# label 395
label395 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-25-Jul-2016-49 A AP.jpg.mat"))["p2"]

label395[63, 1] = 2324

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-25-Jul-2016-49 A AP.jpg.mat"), {"p2": label395})

# label 440
label440 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-26-Jul-2016-72 B AP.jpg.mat"))["p2"]

label440[24, 0] = label440[24, 0] - 7
label440[26, 0] = label440[26, 0] - 7

label440[28, 1] = label440[28, 1] + 2

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-26-Jul-2016-72 B AP.jpg.mat"), {"p2": label440})

# label 446
label446 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-26-Jul-2016-76 C AP.jpg.mat"))["p2"]

label446[42, 0] = label446[42, 0] + 2
label446[42, 1] = label446[42, 1] + 4

label446[29, 1] = label446[29, 1] - 3

for k in range(60):
    label446[k, 0] = label446[k + 8, 0]
    label446[k, 1] = label446[k + 8, 1]

label446[59, 0] = label446[59, 0] - 2

label446[60, 0] = label446[60, 0] + 6
label446[60, 1] = label446[60, 1] + 27

label446[61, 0] = label446[61, 0] + 6
label446[61, 1] = label446[61, 1] + 37

label446[62, 0] = label446[62, 0]
label446[62, 1] = label446[62, 1] + 29

label446[63, 0] = label446[63, 0]
label446[63, 1] = label446[63, 1] + 37

label446[64, 0] = label446[64, 0] - 4
label446[64, 1] = label446[64, 1] + 36

label446[65, 0] = label446[65, 0] - 1
label446[65, 1] = label446[65, 1] + 40

label446[66, 0] = label446[66, 0] - 6
label446[66, 1] = label446[66, 1] + 39

label446[67, 0] = label446[67, 0] - 4
label446[67, 1] = label446[67, 1] + 40

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-26-Jul-2016-76 C AP.jpg.mat"), {"p2": label446})

# label 472
label472 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-29-Dec-2016-156 C AP.jpg.mat"))["p2"]

label472[53, 1] = label472[53, 1] - 3
label472[56, 1] = label472[56, 1] - 3
label472[57, 1] = label472[57, 1] - 4
label472[60, 1] = label472[60, 1] - 3
label472[61, 1] = label472[61, 1] - 3
label472[64, 1] = label472[64, 1] - 2
label472[65, 1] = label472[65, 1] - 2

label472[54, 1] = label472[54, 1] + 3
label472[55, 1] = label472[55, 1] + 3
label472[58, 1] = label472[58, 1] + 3
label472[59, 1] = label472[59, 1] + 3
label472[62, 1] = label472[62, 1] + 3
label472[63, 1] = label472[63, 1] + 3
label472[66, 1] = label472[66, 1] + 3
label472[67, 1] = label472[67, 1] + 3

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-29-Dec-2016-156 C AP.jpg.mat"), {"p2": label472})

# label 475
label475 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-30-Dec-2016-158 A AP.jpg.mat"))["p2"]

for k in range(38, 64):
    label475[k, 0] = label475[k + 4, 0]
    label475[k, 1] = label475[k + 4, 1]

for k in range(54, 66):
    label475[k, 0] = label475[k + 2, 0]
    label475[k, 1] = label475[k + 2, 1]

label475[62, 1] = label475[62, 1] + 18

label475[63, 0] = label475[63, 0] + 4
label475[63, 1] = label475[63, 1] + 20

label475[64, 0] = label475[64, 0] - 2
label475[64, 1] = label475[64, 1] + 18

label475[65, 1] = label475[65, 1] + 20

label475[66, 0] = label475[66, 0] - 2
label475[66, 1] = label475[66, 1] + 30

label475[67, 1] = label475[67, 1] + 32

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-30-Dec-2016-158 A AP.jpg.mat"), {"p2": label475})

# label 477
label477 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-30-Dec-2016-159 A AP2.jpg.mat"))["p2"]

for k in range(48):
    label477[67 - k, 0] = label477[67 - k - 2, 0]
    label477[67 - k, 1] = label477[67 - k - 2, 1]

label477[51, 0] = label477[51, 0] - 3

label477[20, 1] = label477[20, 1] + 2
label477[21, 1] = label477[21, 1] + 2

label477[22, 1] = label477[22, 1] + 2
label477[23, 1] = label477[23, 1] + 2

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-30-Dec-2016-159 A AP2.jpg.mat"), {"p2": label477})

# label 481
label481 = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-30-Dec-2016-161 A AP.jpg.mat"))["p2"]

label481[64, 0] = label481[60, 0]
label481[64, 1] = label481[60, 1]

label481[65, 0] = label481[61, 0]
label481[65, 1] = label481[61, 1]

label481[66, 0] = label481[62, 0]
label481[66, 1] = label481[62, 1]

label481[67, 0] = label481[63, 0]
label481[67, 1] = label481[63, 1]

scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-30-Dec-2016-161 A AP.jpg.mat"), {"p2": label481})

# last 4 landmarks at the top of spine for all of the following images
label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-212 A AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-09-Jan-2017-212 A AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-227 C AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-227 C AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-227 E AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-227 E AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-228 B AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-228 B AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-230 A AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-230 A AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-237 B AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-237 B AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-237 D AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-237 D AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-243 C AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-10-Jan-2017-243 C AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 B AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 B AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 C AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 C AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 D AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 D AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 E AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-11-Jan-2017-255 E AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-12-Jan-2017-258 A AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-12-Jan-2017-258 A AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-20-Jan-2017-275 G AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-20-Jan-2017-275 G AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-23-Jan-2017-276 B AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-23-Jan-2017-276 B AP.jpg.mat"), {"p2": label})

label = scipy.io.loadmat(os.path.join(train_label_dir, "sunhl-1th-25-Jan-2017-279 B AP.jpg.mat"))["p2"]
label[:, :] = np.roll(label[:, :], 4, axis=0)
scipy.io.savemat(os.path.join(train_label_dir, "sunhl-1th-25-Jan-2017-279 B AP.jpg.mat"), {"p2": label})
