# VGG
Keras implementation of VGG from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## Insturctions
1. Clone the repository and navigate to the downloaded folder.
```
	git clone https://github.com/MrinalJain17/VGG.git
	cd VGG
```

2. Import the module and get the required model.
```python
	from vgg import VGG
	
	# Getting VGG-16 (16-layer VGG model) for a dataset with 20 classes
	model = VGG(model_type='D', num_classes=20)
	model.summary()
```

3. The model was tested on the cifar-10 dataset.  
For further details, view the jupyter notebook by running the command:
```bash
	jupyter notebook cifar-10.py
```

## Requirements
`Python 3.x` (preferably from the [Anaconda Distribution](https://www.anaconda.com/download/))

### Additional Libraries to install
- [Keras]()  
```bash
	pip install keras
```
 (With backend as [Tensorflow](https://www.tensorflow.org/))  
 For installation of Tensorflow, view instructions on their site [here](https://www.tensorflow.org/install/).
