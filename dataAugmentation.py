import os
import cv2.cv2 as cv2
import numpy as np
import scipy.io
import pickle
import csv
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# desired image width and height
img_width = 256
img_height = 512



# Input data directories
train_data_dir = "dataset/data/training"
train_label_dir = "dataset/labels/training"
train_label_csv_name = "/datasetlabels/training/modified_train_landmarks.csv"
train_filename_dir = "dataset/labels/training/modified_train_filenames.csv"
# Output directories
train_augmented_data_dir = "dataset/data/augmentedTraining"
train_augmented_label_dir = "dataset/labels/augmentedTraining"

# Data Augmentation
#   - mirroring
#   - tilting (small angles only)
#   - adjusting gamma
#   Expand the data set, not considering large angle tilting because it won't happens in real life.


filenames_lst = open(train_filename_dir, 'r').read().split("\n")[:-1]

landmark = list(csv.reader(open(train_label_csv_name)))
for i in range(len(landmark)):
  for j in range(136):
    landmark[i][j] = float(landmark[i][j])

index = -1
sum_vert = 0
sum_conn = 0
for filename in filenames_lst:
    index += 1
    curr_label = landmark[index]
    corresponding_label = np.asarray([curr_label[:68], curr_label[68:]]).T
    corresponding_label[:, 0] = corresponding_label[:, 0] * img_width
    corresponding_label[:, 1] = corresponding_label[:, 1] * img_height


    # print("Preprocessing: " + filename + " ...")

    corresponding_data_dir = os.path.join(train_data_dir, filename)
    corresponding_label_dir = os.path.join(train_label_dir, filename)

    extracted_image = cv2.imread(corresponding_data_dir, cv2.IMREAD_GRAYSCALE)
    corresponding_image = cv2.resize(extracted_image, (img_width, img_height))

    horizontal_ratio = img_width / extracted_image.shape[1]
    vertical_ratio = img_height / extracted_image.shape[0]
    corresponding_label = scipy.io.loadmat(corresponding_label_dir)["p2"]

    corresponding_label[:, 0] = corresponding_label[:, 0].astype("float") * horizontal_ratio
    corresponding_label[:, 1] = corresponding_label[:, 1].astype("float") * vertical_ratio
    print(corresponding_label)

    # # visualization of labels in the input image
    # visualization_data = np.copy(corresponding_image)
    # for landmark in corresponding_label:
    #     visualization_data[landmark[1]][landmark[0]] = 255
    # cv2.imshow("visualization", visualization_data)
    # cv2.waitKey(0)

    # store original image and label to output directory
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename), corresponding_image)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]), corresponding_label)

    # --------------------------------adjusting gamma (landmark locations not changed)-------------------------------- #
    gamma_image_a = adjust_gamma(corresponding_image, 0.5)
    gamma_image_b = adjust_gamma(corresponding_image, 1.3)

    store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " gamma adjusted A.jpg", gamma_image_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " gamma adjusted A", corresponding_label)

    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " gamma adjusted B.jpg", gamma_image_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " gamma adjusted B", corresponding_label)

    # ------------------------------------mirroring (landmark locations mirrored)------------------------------------ #
    mirrored_image = cv2.flip(corresponding_image, 1)

    im_width = corresponding_image.shape[1]
    mirrored_label = np.asarray(np.concatenate(
        (np.matrix(im_width - corresponding_label[:, 0]).T, np.matrix(corresponding_label[:, 1]).T), axis=1))

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " mirrored.jpg", mirrored_image)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " mirrored", mirrored_label)

    # adjusting gamma for mirrored samples
    gamma_mirrored_image_a = adjust_gamma(mirrored_image, 0.5)
    gamma_mirrored_image_b = adjust_gamma(mirrored_image, 1.3)

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " mirrored gamma adjusted A.jpg",
                gamma_mirrored_image_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " mirrored gamma adjusted A", mirrored_label)

    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " mirrored gamma adjusted B.jpg",
                gamma_mirrored_image_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " mirrored gamma adjusted B", mirrored_label)

    # -------------------------------------- tilting (landmark locations tilted)-------------------------------------- #
    img_center = (corresponding_image.shape[1] / 2, corresponding_image.shape[0] / 2)
    label_add_one_transposed = np.append(corresponding_label.T, [np.ones(corresponding_label.shape[0])], axis=0)

    rotation_matrix_a = cv2.getRotationMatrix2D(img_center, np.random.randint(-5, 0), 1)
    rotation_matrix_b = cv2.getRotationMatrix2D(img_center, np.random.randint(1, 6), 1)

    tilted_image_a = cv2.warpAffine(corresponding_image, rotation_matrix_a,
                                    (corresponding_image.shape[1], corresponding_image.shape[0]))
    tilted_image_b = cv2.warpAffine(corresponding_image, rotation_matrix_b,
                                    (corresponding_image.shape[1], corresponding_image.shape[0]))

    tilted_label_a = np.floor(np.dot(rotation_matrix_a, label_add_one_transposed).T).astype(int)
    tilted_label_b = np.floor(np.dot(rotation_matrix_b, label_add_one_transposed).T).astype(int)

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted A.jpg", tilted_image_a)
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted B.jpg", tilted_image_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted A", tilted_label_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted B", tilted_label_b)

    # adjusting gamma for tilted samples
    gamma_tilted_image_a_a = adjust_gamma(tilted_image_a, np.random.randint(15, 21) / 10)
    gamma_tilted_image_a_b = adjust_gamma(tilted_image_a, 1.3)

    gamma_tilted_image_b_a = adjust_gamma(tilted_image_b, np.random.randint(15, 21) / 10)
    gamma_tilted_image_b_b = adjust_gamma(tilted_image_b, np.random.randint(21, 27) / 10)

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted A gamma adjusted A.jpg",
                gamma_tilted_image_a_a)
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted A gamma adjusted B.jpg",
                gamma_tilted_image_a_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted A gamma adjusted A", tilted_label_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted A gamma adjusted B", tilted_label_a)

    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted B gamma adjusted A.jpg",
                gamma_tilted_image_b_a)
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted B gamma adjusted B.jpg",
                gamma_tilted_image_b_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted B gamma adjusted A", tilted_label_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted B gamma adjusted B", tilted_label_b)

    # ---------------------------------------------tilting and mirroring--------------------------------------------- #
    mirrored_label_add_one_transposed = np.append(mirrored_label.T, [np.ones(corresponding_label.shape[0])], axis=0)

    rotation_matrix_c = cv2.getRotationMatrix2D(img_center, np.random.randint(-5, 0), 1)
    rotation_matrix_d = cv2.getRotationMatrix2D(img_center, np.random.randint(1, 6), 1)

    tilted_mirrored_image_a = cv2.warpAffine(mirrored_image, rotation_matrix_c,
                                             (corresponding_image.shape[1], corresponding_image.shape[0]))
    tilted_mirrored_image_b = cv2.warpAffine(mirrored_image, rotation_matrix_d,
                                             (corresponding_image.shape[1], corresponding_image.shape[0]))

    tilted_mirrored_label_a = np.floor(np.dot(rotation_matrix_c, mirrored_label_add_one_transposed).T).astype(int)
    tilted_mirrored_label_b = np.floor(np.dot(rotation_matrix_d, mirrored_label_add_one_transposed).T).astype(int)

    store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted mirrored A.jpg",
                tilted_mirrored_image_a)
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted mirrored B.jpg",
                tilted_mirrored_image_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted mirrored A", tilted_mirrored_label_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted mirrored B", tilted_mirrored_label_b)

    # adjusting gamma for mirrored and titled samples
    gamma_tilted_mirrored_image_a_a = adjust_gamma(tilted_mirrored_image_a, np.random.randint(15, 21) / 10)
    gamma_tilted_mirrored_image_a_b = adjust_gamma(tilted_mirrored_image_a, np.random.randint(15, 21) / 10)

    gamma_tilted_mirrored_image_b_a = adjust_gamma(tilted_mirrored_image_b, np.random.randint(15, 21) / 10)
    gamma_tilted_mirrored_image_b_b = adjust_gamma(tilted_mirrored_image_b, np.random.randint(15, 21) / 10)

    store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted mirrored A gamma adjusted A.jpg",
                gamma_tilted_mirrored_image_a_a)
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted mirrored A gamma adjusted B.jpg",
                gamma_tilted_mirrored_image_a_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted mirrored A gamma adjusted A",
            tilted_mirrored_label_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted mirrored A gamma adjusted B",
            tilted_mirrored_label_a)

    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted mirrored B gamma adjusted A.jpg",
                gamma_tilted_mirrored_image_b_a)
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted mirrored B gamma adjusted B.jpg",
                gamma_tilted_mirrored_image_b_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted mirrored B gamma adjusted A",
            tilted_mirrored_label_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted mirrored B gamma adjusted B",
            tilted_mirrored_label_b)
