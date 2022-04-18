# CSC2621FinalProject
Code and Documentation for CSC2621 final project

Dataset can be obtained from http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets, dataset 16<br />
Preprocessing is included in the code, so no extra preprocessing to the dataset is needed.<br />

Cite dataset:
Wu, H., Bailey, Chris., Rasoulinejad, Parham., and Li, S., 2017.<br />
  Automatic landmark estimation for adolescent idiopathic scoliosis assessment using boostnet.<br />
  Medical Image Computing and Computer Assisted Intervention:127-135.
  
Preperation:
  1. Upload dataset to your google drive<br />
  2. Change all the file paths in CSC2621Project.ipnb<br />

Results can be reproduced by running all cells in order, outputs may be different each time. 
Responsibility of each cell:
  First cell: mount google drive, so you can get access to files in your google drive<br />
  Second cell: import all packages that will be used in this project<br />
  Third cell: declare global variables<br />
  Fourth cell: data preprocessing and prepare ground truth angles file for training and testing purposes. New files are stored in google drive<br />
  Fifth cell: load training and test data to torch dataloader, get ready for training<br />
  Sixth cell: build CNN architecture and declare loss function<br />
  Seventh cell: training begins<br />
  Eighth cell: store model to google drive after training completed. Compute test set results and also store them to google drive<br />
  Ninth cell: compute and print SMAPE + other statistics from the test set results and ground truth information
