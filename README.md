# MRI Super-Resolution

This is the code repository for the paper "Perceptual cGAN for MRI Super-Resolution". The model proposed in the paper was trained and tested on [SuperMUDI Challenge](http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html) dataset.


## Requirements
Please install the dependencies listed in requirements.txt
```
pip install -r requirements.txt
```

In order to run the code, please follow the steps below.

## Processing Dataset
After downloading the dataset from the challenge website, preprocess and create numpy arrays.

## Training 
Set the training options in the file `options.py` or provide them via command line.
Start training using the following command:
```
python train.py
```

## Testing
Run the script `inference.py` for running inference and getting the test scores
```
python inference.py
```

##Citation:
If you find this code useful kindly cite our work:-)
Nasser, S.A., Shamsi, S., Bundele, V., Garg, B. and Sethi, A., 2022. Perceptual cGAN for MRI Super-resolution. arXiv preprint arXiv:2201.09314.

@article{nasser2022perceptual,
  title={Perceptual cGAN for MRI Super-resolution},
  author={Nasser, Sahar Almahfouz and Shamsi, Saqib and Bundele, Valay and Garg, Bhavesh and Sethi, Amit},
  journal={arXiv preprint arXiv:2201.09314},
  year={2022}
}
