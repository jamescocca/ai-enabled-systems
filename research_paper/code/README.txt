NOTE: The local_analysis.py, settings.py, and main.py files are from the ProtoPNet repository and have been included to show my modifications that I made to the ProtoPNet code in order to train the model on the gender classification dataset. All of my changes are noted with comments on the lines before and after the changes. To run the model, these would need to be pasted into the full repository at: https://github.com/cfchen-duke/ProtoPNet

The model.zip file contains compressed files of the trained model. The .pth file is the trained model, and the compressed img folder contains the generated prototype images for that model, which are necessary for performing prototype analysis with the resulting model file.

The prototype_analysis.zip folder contains sub-directories of light-skinned and dark-skinned male and female samples that have been fed through the network in order to generate prototype analysis output via the local_analysis.py file. This must be unzipped to run the Jupyter Notebook.

The notebook, prototype_analysis_viewer.ipynb, contains visualizations for a random sample for light-skinned and dark-skinned males and females in order to demonstrate the most activated section of that sample with the highest matching prototype. This file has already been executed for viewing purposes, but can be re-run if desired to generate further random output.

Not included here is the gender classification dataset, which must be pulled and unzipped from: https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset