- The complete set of x-ray images can be found in the folder 'data'.
The filename includes an unique image id, an anonymized patient ID, left or right breast and the view. For instance, in the filename "20586908_6c613a14b80a8591_MG_R_CC_ANON", "2056908" is the unique image id, "6c613a14b80a8591" is the patient ID that can be used to aggregate all 4 views of images, "R" stands for the breast and "CC" stands for the view.

- train.txt and val.txt is the file list of training and validation data samples
each line stands for a data sample. The unique image id and the label is splitted by '\t'

-test.txt is the file list of test data samples
each line stands for a test data sample. It contains the unique image id only.
