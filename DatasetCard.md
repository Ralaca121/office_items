
| Feature Name | DataType           |Status         | Total |
| :---         | :---               | :---          | :---  |
| Bin          | .jpg / .jpeg/ .png | Fully Labeled |100    |
| Book         | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| Chair        | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| Computer     | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| Pens         | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| Stapler      | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| Table        | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| Telephone    | .jpg / .jpeg/ .png | Fully Labeled | 100   |
| None         | .jpg / .jpeg/ .png | Unlabeled     | 90    |
| Total | 890 | 800 Labeled | 

Intended Task : Object detection to sort office items

Data Source : Google Image

Data Split : Train/Test/Val ; 73/13/14

### Annotation Status
Every single image belonging to the 8 target classes is **100% labeled** (800 images total). The remaining 90 images are explicit background/negative samples containing none of the target classes, included to minimize false positives during model training.
The images was labeled and split using Roboflow
