This code release accompanies the paper [1] and is based on the
"Regression Tree Fields" [2, 3, 4] source code of Microsoft
Research. Please find included the license terms and conditions.


Getting started
===============

Compile Deblur/Deblur.cpp so that the binary resides in build/Deblur.
Running it will write output to demo/predictions/<Cascade_Depth>. The
demo/images folder contains as input the first benchmark image of
Levin et al. [5], the rest is available online. Input image names must
be listed in demo/test.txt. The initial kernel estimates used for the
results reported in the paper are located in demo/initial.

To test on different data, place the blurry image in the demo/images
folder and the initial blur estimate in demo/initial following the
naming convention <input_image_name>_kernel.dlm.


Further notes
=============

Interleaved blur kernel estimation is activated by calling the
Interleaved method of the Dataset object as demonstrated in
Deblur.cpp. Note that the optimal number of cascade levels differs for
interleaved and standard evaluation.

The models folder contains both the interleaved and standard learned
RTF models used in the paper.

The file kernels.mat contains the recorded LED trajectories of human
camera shakes as explained in the paper.


----------------------------------------------------------------------

[1] Kevin Schelten, Sebastian Nowozin, Jeremy Jancsary, Carsten
Rother, and Stefan Roth. Interleaved regression tree field cascades
for blind image deconvolution. In WACV 2015.

[2] Jeremy Jancsary, Sebastian Nowozin, Toby Sharp, and Carsten
Rother. Regression tree fields - an efficient, non-parametric approach
to image labeling problems. In CVPR 2012.

[3] Jeremy Jancsary, Sebastian Nowozin, and Carsten
Rother. Loss-specific training of non-parametric image restoration
models: A new state of the art. In ECCV 2012.

[4] Jeremy Jancsary, Sebastian Nowozin, and Carsten Rother. Learning
Convex QP Relaxations for Structured Prediction. In ICML 2013.

[5] Anat Levin, Yair Weiss, Fredo Durand, and William
T. Freeman. Understanding and evaluating blind deconvolution
algorithms. In CVPR 2009.
