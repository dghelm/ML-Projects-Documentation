# Machine Learning Projects

This repo is devoted to cataloging the machine learning projects I've developed. Accumulating the code from various old computers and collaborators is a work in progress, but document explains a timeline of the work.

## 2016

### Austism Spectrum Disorder Detection using fMRI Scans

Developed as a semester-long project for CS5033 at the University of Oklahoma, this project used a publically available dataset of functional MRIs scans for a group a patients. At the time, most ML libraries lacked support 3D CNNs, and this dataset was both 3-dimensional and a time-series. To work around this, I tested various methods of projecting this "4D" imagery into 2D space to be used with well-established imaging-based architectures (via transfer learning), along with a few other architectures including convolutional autoencoders.

Code has not been cleaned or consolidated, but was a loose set of Jupiter Notebooks sharing helper modules that eventually incorporated Tensorboard for additional imaging and training metrics.

_Final Paper available [here]('./01-AutismFMRI/AutismFMRI-Paper.pdf')._

_Slideshow of final presentation is available [here]('./01-AutismFMRI/AutismFMRI-Presentation.pdf')._

## 2017

### LORISAL

Developed as an honors project while an undergraduate in the Computer Science program at the University of Oklahoma, LORISAL was a tool to take an online archive of scanned books, scrape the images, arrange them in a database, do OCR and image identification to try and ultimately generate captions and create a search engine for texts based on the images it contained.

The project had various modules adjacent to ML processes (including OpenCV usage), but utilized an image labeling model "img2txt" built and trained in TensorFlow.

Ultimately, the project was using texts from the library's History of Science collection, so the graphics were not a good fit for the model that had been trained on photographs and the semester ended before developing a solution for developing appropriate datasets for re-training the model.

_[Github repo] (https://github.com/dhelma/lorisal)_

_The project was also presented to the OKC Python User Group as a way of introducing the processes and libraries involved, which can be viewed [here] ('./02-LORISAL/LORISAL-PythonUserGroup.pdf')._

## 2018

### Flowrate Estimation for Sensor Readings for Remote Well Monitoring

Developed for CS5043 at the University of Oklahoma and in collaboration with another student and a real-world data provided by a local remote monitoring company, this project aimed to estimate flowrate using time-series data collected from other monitoring sensors on natural gas wells. This was my first deep-dive on time-series data, GRU networks and LSTM networks.

_Code for this is on an old machine, but I will need to sanitize customer-identifying information before publishing to GitHub._

_Presentation Poster Available [here] ('./03-FlowRateEstimation/FlowRateEstimation-Poster.pdf')._

## 2019 & 2020

*The following projects were build while in OU's School of Visual Art, pursuing a Master of Fine Arts in their Art, Technology and Culture department. I was using trained machine learning models (mostly image synthesizing GANs) as a material to start from and only the final project involves me training my own network.*

### Arch by AI Show

#### Generative Landscape: Churches

A video work stitching panoramas generated in real-time from pre-trained image synthesizing model.

_More info on my [portfolio] (https://dgh.works/works/generative-landscape-churches/)._

#### Generative Towers VR Space

A project allowing the user to "collaborate" with the generative model by blending synthesized images in an immersive VR environment. This works by taking from a stating palette of pre-rendered images generated from random vectors. Then, by choosing which images (and their associated vectors) to blend, the program calculates weighted-averages of the vector from the model's latent space and renders them in real-time. The user can then save the results for later use or publish them to social media.

_More info on my [portfolio] (https://dgh.works/works/generative-towers-vr-space/)._

### State Machine

#### 30 Under 30

A 3-channel video piece using images of local award-recipients. The project used a model find a latent closest to representing the image in a pre-trained model, and then animated movement in the latent-space between award winners.

_Video sample [here] (https://drive.google.com/file/d/12Z1bCFUtZJNPrYjgQ_M0K_f9sSV-t9l9/view?usp=sharing)._

#### Political Ads

This work involved collaborating with a local archive of political advertisments to use Google Cloud and Compute services to transcribe over 6000 Oklahoma political ads from across 6 decades. My software then trained a model to generate its own "political ad" sentences, which were then connected to the closest actually-existing videoclips and edited together into a 6 hour, 2 channel video.

_Code for this work is on another machine, but was developed in a series of notebooks, which should be posted here shortly._

_Video sample [here] (https://drive.google.com/file/d/1mWVYybDJWivteDGWbAvZh7VG0qIj27cY/view?usp=sharing)._
