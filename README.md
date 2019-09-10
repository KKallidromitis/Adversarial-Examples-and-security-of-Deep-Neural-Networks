# Adversarial-Examples-on-Neural-Networks
Bachelor of Engineering thesis
I was always fascinated how deep learning was so "intelligent" and could outperform all machine learning algorithms. So for my thesis I decided to embarass the cutting edge deep learning networks by applying attacks and observing how easy it is to trick them compared to a human.

Adversarial attacks are generated on ResNet Neural networks in PyTorch and defenses to counteract the attacks. Example datasets used is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/). The adversarial attacks are aimed to cause missclassification in the traffic signs or to lower the cosine similarity between faces of the same person. This, in turn, will cause a failure in the convolutional network which can have dangerous consequences in the real world.

*  Model file ->classification model and the data preprocessing
*  adv_generation file -> adversarial attacks such as EoT, Graffiti attacks, salt and pepper, etc
*  adv_defences file -> adversarial defences such as defensive distillation, denoising models, etc
