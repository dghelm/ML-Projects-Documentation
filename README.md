# Machine Learning Projects

This repo is devoted to cataloging the machine learning projects I've developed. Accumulating the code from various old computers and collaborators is a work in progress, but document explains a timeline of the work.

## 2016

### Austism Spectrum Disorder Detection using fMRI Scans

Developed as a semester-long project for CS5033 at the University of Oklahoma, this project used a publically available dataset of functional MRIs scans for a group a patients. At the time, most ML libraries lacked support 3D CNNs, and this dataset was both 3-dimensional and a time-series. To work around this, I tested various methods of projecting this "4D" imagery into 2D space to be used with well-established imaging-based architectures (via transfer learning), along with a few other architectures including convolutional autoencoders.

Code has not been cleaned or consolidated, but was a loose set of Jupiter Notebooks sharing helper modules that eventually incorporated Tensorboard for additional imaging and training metrics.

_Final Paper available [here]('./01-AutismFMRI/AutismFMRI-Paper.pdf')._

_Slideshow of final presentation is available [here]('./01-AutismFMRI/AutismFMRI-Presentation.pdf')._
