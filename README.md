# Simpsons Classifier

Training a deep conv. neural network to recognize characters from the classic TV show _The Simpsons_. This a simple toy example for a multiclass classification problem written in Python and PyTorch. 

## Training Data

The training and test for the classification problem is available from as a download from Kaggle https://www.kaggle.com/alexattia/the-simpsons-characters-dataset.

To run this experiment yourself, download the files and extract the data using this folder scheme with one folder for each class/character. 

```
.
└── data
    ├── train {for all training images}
          ├── homer
          ├── lisa
          ├── ...
    └── test {for all test images}
          ├── homer
          ├── lisa 
          ├── ...
```

Some example images and their labels from the training set:

![Training Data Overview](https://i.imgur.com/SFo99Zl.png)

## Results

The code for this classification problem is provided in the `main.ipynb` Jupyter notebook.  

After training the deep neural network for 20 epochs it reaches an accuracy of __91%__ on the test dataset. Loss and accuracy curves over the train duration:

![Loss and Accuracy](https://i.imgur.com/vWtKCrX.png)

Apart from some characters that look similar (e.g. Patty and Selma) the network is able to accuratly recognise most characters on unseen test images. Confussion matrix of the classification results on the test dataset: 

![Confussion Matrix](https://i.imgur.com/tZ71RiP.png)



## If you want to run and train this yourself:

1. Install the required modules:

`$ pip3 install numpy matplotlib seaborn torch torchvision jupyterlab`

2. Clone the repository:

```
$ git clone https://github.com/paddepadde/simpsons-classification.git
$ cd simpsons-classification
```

3. Run the notebook `main.ipynb` included in the repo:

`$ jupyer notebook`

