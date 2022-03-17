# ES-CCGAN, the implementation of the paper of "Unsupervised Haze Removal for High-Resolution Optical Remote-Sensing Images Based on Improved Generative Adversarial Networks", the link of this paper is "https://www.mdpi.com/2072-4292/12/24/4162".
This is a remote sensing image dehazing code, and this is realized by python.
To run this project you need to set up the environment, download the dataset, run a script to process data, and then you can train and test the network models. 
I will show you step by step to run this project and I hope it is clear enough.

--Prerequisite
I tested my project in Intel Core i9, 64G RAM, GPU RTX 2080 Ti. Because it takes about several days for training, I recommend you using CPU/GPU strong enough and about 24G Video Memory.

--Dataset
I use a self-made remote sensing image which consists of 52376 haze-free images, 52376 hazy images, and 52376 haze-free images with blurred edges. All the images were 256 × 256 pixels in size. All of the data need to transform to tfrecords. The code of generated haze remote sensing image is in the "ImageFogger.py" 

--Training
To train a generator, run the following command
python train.py

--Test
First, the model needs to transform to  the type of '.pb', run the following command
python export_graph.py
Second, the haze image is dehazed as following:
python inference.py
