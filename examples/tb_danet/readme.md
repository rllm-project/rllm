### To use

```
python train.py -c .\config\xxxx.yaml -g number_x
```
`xxxx` is the name of the dataset being used, and `number_x` is the id of GPU.


### Data preprocessing

using `svm2pkl()` to transform train.txt, vali.txt, test.txt in .\data\dataset\MSRL-WEB10K\Fold1 into .csv files, using `csv2pkl()` to transform train.csv, valid.csv, test.csv in .\dataset\movie_cla\movies into .pkl files.