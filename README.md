# CSC2621FinalProject
Code and Documentation for CSC2621 final project

Dataset can be obtained from http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets, dataset 16<br />
Data augmentation is in a separate file called dataAugmentation.py<br />
Preprocessing (except data augmentation) is included in the code, so no extra preprocessing to the dataset is needed.<br />

Cite dataset:
Wu, H., Bailey, Chris., Rasoulinejad, Parham., and Li, S., 2017.<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Automatic landmark estimation for adolescent idiopathic scoliosis assessment using boostnet.<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Medical Image Computing and Computer Assisted Intervention:127-135.
  
Preperation:
  1. Run fixDataLandmarkError.py to correct errors in the GT landmark file<br />
  2. Run dataAugmentation.py in local machine to augment the data<br />
  3. Upload augmented dataset to your google drive<br />
  4. Change all the file paths in CSC2621Project.ipnb<br />

Results can be reproduced by running all cells in order, outputs may be different each time. 
Responsibility of each cell:<br />
  &nbsp;&nbsp;&nbsp;&nbsp;First cell: mount google drive, so you can get access to files in your google drive<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Second cell: import all packages that will be used in this project<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Third cell: declare global variables<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Fourth cell: data preprocessing and prepare ground truth angles file for training and testing purposes. New files are stored in google drive<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Fifth cell: load training and test data to torch dataloader, get ready for training<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Sixth cell: build CNN architecture and declare loss function<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Seventh cell: training begins<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Eighth cell: store model to google drive after training completed. Compute test set results and also store them to google drive<br />
  &nbsp;&nbsp;&nbsp;&nbsp;Ninth cell: compute and print SMAPE + other statistics from the test set results and ground truth information
