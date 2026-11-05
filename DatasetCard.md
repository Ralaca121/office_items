* Dataset Name: Office Items Object Detection Dataset 
* Task Type: Single & Multi-Object Detection (Bounding Box)
* Image Resolution: 640 x 640 pixels
* Negative Background Samples: 90 empty office/desk scenes (0 bounding boxes)

| Feature Name | DataType           |Status         | 
| :---         | :---               | :---          |
| Bin          | .jpg / .jpeg/ .png | Fully Labeled | 
| Book         | .jpg / .jpeg/ .png | Fully Labeled | 
| Chair        | .jpg / .jpeg/ .png | Fully Labeled | 
| Computer     | .jpg / .jpeg/ .png | Fully Labeled | 
| Pen          | .jpg / .jpeg/ .png | Fully Labeled | 
| Stapler      | .jpg / .jpeg/ .png | Fully Labeled |
| Table        | .jpg / .jpeg/ .png | Fully Labeled | 
| Telephone    | .jpg / .jpeg/ .png | Fully Labeled | 
| None         | .jpg / .jpeg/ .png | Unlabeled     | 

| Feature Name | Train   |Test  | Val | Total |
| :---         | :---    | :--- | :---|       |
| Bin          | 70      | 15   | 10  |  95   |
| Book         | 66      | 11   | 13  |  90   |
| Chair        | 67      | 12   | 13  |  92   |
| Computer     | 70      | 12   | 11  |  93   |
| Pen          | 67      | 14   | 15  |  96   |
| Stapler      | 70      | 13   | 14  |  97   |
| Table        | 67      | 15   | 14  |  96   |
| Telephone    | 70      | 15   | 15  |  100  |
| None         | 66      | 12   | 12  |  90   |
| Total | 851  | 761 Labeled | 

Intended Task : Object detection to sort office items

Data Source : Google Image

Data Split : Train/Test/Val ; 73/13/14

### Annotation Status
Every single image belonging to the 8 target classes is **100% labeled** (800 images total). The remaining 90 images are explicit background/negative samples containing none of the target classes, included to minimize false positives during model training.
The images was labeled and split using Roboflow
