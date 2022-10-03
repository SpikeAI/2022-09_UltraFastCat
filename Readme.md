# Ultra-fast categorization of image containing animals *in vivo* and *in computo*

Animals are able to categorize images from broad categories in a very efficient and rapid fashion. Humans, for instance, can detect the presence of an animal in an image in as little as 120 ms. In the last decade, the field of artificial intelligence has experienced one remarkable breakthrough. In a relatively short period of time, neuroscientifically-inspired deep-learning algorithms designed to perform a visual recognition task literally bloomed.  Artificial networks now achieve human-like performance levels, but are usually trained on less ecological tasks, for instance the 1000 categories of the ImageNet challenge. Here, we retrained the VGG Convolutional Neural Network adapted to ImageNet on two ecological tasks : detecting animals or artefacts in the image. We show that retraining the network achieves human-like performance level and we could also reproduce the accuracy of the detection on an image-by-image basis. This showed in particular that these two tasks perform better if combined as animals (e.g. lions) tend to be less present in photographs containing artefacts (e.g. buildings). Then, we reproduce some behavioural observations from humans such as the robustness to rotations (e.g. upside-down image). Finally, we could test the number of layers of the CNN which are necessary to reach such a performance, showing that a good accuracy for ultra-fast categorization could be reached with a few layers. We expect to apply this network to perform model-based experiments.

![An ultra fast cat](https://www.funny-games.biz/images/pictures/1922-ultra-fast-cat.jpg)


* read the current preprint on [arxiv](https://arxiv.org/abs/2205.03635)
* check-out the associated [zotero group](https://www.zotero.org/groups/4560566/ultrafastcat)

## installation

### dataset

Fetch all images for the three tasks: 'animal', 'artifact' and 'random':

```commandline
git clone https://github.com/SpikeAI/DataSetMaker
cd DataSetMaker
python dataset_synset.py
cd ..
```

### download the notebook

```commandline
git clone https://github.com/SpikeAI/2022-09_UltraFastCat
jupyter notebook
```

### Keywords

categorization, vision, convolutional neural networks, psychophysics, primary visual cortex