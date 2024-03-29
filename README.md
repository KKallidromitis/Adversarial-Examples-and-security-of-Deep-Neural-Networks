# Adversarial Examples and security of Deep Neural Networks

I was always fascinated how deep learning was so "intelligent" and could outperform all machine learning algorithms. So for my thesis I decided to embarass the cutting edge deep learning networks by applying attacks and observing how they compare against a human.

<p align="center"><b> Sample of high confidence misclassification (over 99%) </b></p>
<p align="center">
  <img src="https://github.com/KKarrasKallidromitis/Adversarial-Examples-on-Neural-Networks/blob/master/misclassification_sample.png" width="700">
</p>

Adversarial attacks are generated on ResNet Neural networks in PyTorch and defenses to counteract the attacks. Example datasets used is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/). The adversarial attacks are aimed to cause missclassification in the traffic signs or to lower the cosine similarity between faces of the same person. This, in turn, will cause a failure in the convolutional network which can have dangerous consequences in the real world.

*  **model file** -> classification model and the data preprocessing
*  **adv_generation file** -> adversarial attacks such as EoT, Graffiti attacks, salt and pepper, etc
*  **adv_defences file** -> adversarial defences such as defensive distillation, denoising models, etc

<p align="center"><b> Example of attacks that are invisible to human observer </b></p>
<p align="center">
  <img src="https://github.com/KKarrasKallidromitis/Adversarial-Examples-on-Neural-Networks/blob/master/invisgauss.PNG" width="700">
</p>

## Graffiti Attacks on German Traffic Sign Dataset
The conclusion from our experiments was that an adversary can easily cause severe issues in security-critical systems that must perform well at all times. Its important to study the impact of adversarial attacks on video, to examine if different angles across frames, either from a vehicle or camera, are amore robust to missclasifications.


| Type        | Size  | Portion of Image | Colours     | # of parts| Total Accuracy (%) |
|-------------|-------|------------------|-------------|-----------|--------------------|
| Base Model  | -     |  -               | -           | -         |        94.82       |
| Attack 1    | 900   |  7.2%            | Black       | 3         |        2.61        |
| Attack 2    | 400   |  3.2%            | White       | 1         |        57.42       |
| Attack 3    | 500   |  4.0%            | Black/White | 4         |        7.44        |
