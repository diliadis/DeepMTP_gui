<p align="center"><img src="https://raw.githubusercontent.com/diliadis/DeepMTP_gui/main/images/deepMTP_logo_plus_streamlit_logo_white.png#gh-dark-mode-only" alt="logo" height="250"/></p>
<p align="center"><img src="https://raw.githubusercontent.com/diliadis/DeepMTP_gui/main/images/deepMTP_logo_plus_streamlit_logo.png#gh-light-mode-only" alt="logo" height="250"/></p>

<h3 align="center">
<p> A Streamlit app for DeepMTP </h3>

This is the official repository for the gui application of DeepMTP, a deep learning framework. The application uses streamlit for the implementation of the app and [DeepMTP](https://github.com/diliadis/DeepMTP) for the backend. The user can do the following:
* Load one of the built-in datasets or upload their own
* Define the ranges of the hyperparameters of DeepMTP
* Select one of the hyperparameter optimization methods (HPO) methods available
* Select one of the loggin options that are available
* Train the network and check basic info

### Latest Updates
- [15/6/2022] The first implementation of DeepMTP_gui is now live!!!


# Installing DeepMTP_gui
The application uses DeepMTP in the backend, so the use of a GPU is strongly recommended. The installation can be done in the following ways:

## Installing on Google Colab
```bash
will be added soon...
```

## Installing from Source

```bash
# download from the github repository
git clone https://github.com/diliadis/DeepMTP_gui.git
cd DeepMTP_gui
conda env create -f environment.yml
conda activate deepMTP_streamlit_env

# start-up the app 
# Replace the port_number with your preferred port (for example 8501)
# if your dataset is unable to load because of the size, you can increase the server.maxUploadSize accordingly
streamlit run main.py --server.port port_number --server.maxUploadSize=2028
```

# What is DeepMTP?
For a detailed explanation of DeepMTP you can visit the official [GitHub repository](https://github.com/diliadis/DeepMTP). The repository explains the theory behind multi-target prediction, the neural network architecture that DeepMTP utilizes, as well as various demo notebooks that show how to use the framework. The DeepMTP_gui repository can be seen as an attempt to offer the basic functionality of DeepMTP to users that don't have programming experience but are interested in machine learning and the area of MTP. 

# Using DeepMTP_gui?
The current version of this application utilizes a recently added streamlit feature, multi-page support. This enables a clearer separation of the different steps needed to successfully train a neural network. In the following sections we will present the different pages that are available in the UI, and show a typical workflow.

## main
<p align="center">
  <img src="https://imgur.com/oIzY0nA.gif" alt="animated" />
</p>
The main page contains the basic information about multi-target prediction, the different settings and validation strategies that are possible.

## DeepMTP architecture

<p align="center">
  <img src="https://imgur.com/8LUTL6L.gif" alt="animated" />
</p>
The "DeepMTP architecture page" contains basic information about the neural network architecture DeepMTP uses



## load dataset

<p align="center">
  <img src="https://imgur.com/ppM4LxG.gif" alt="animated" />
</p>
The "load dataset" page is the first step in the typical workflow of the app. This is the input point for the user, as they can either use one of the built-in datasets offered by the DeepMTP package, or upload their own.

## Use a build-in dataset
When this option is selected, the UI shows two select boxes. The first one displays the MTP problem settings and the second the specific datasets that are available, given a MTP problem setting. The current version of the app supports the following datasets:
|  Function  | Description |
| :--- | :--- |
| `multi-label classification` | the user can load the multi-label classification datasets available in the [MULAN repository](http://mulan.sourceforge.net/datasets-mlc.html). |
| `multivariate regression` | the user can load the multivariate regression datasets available in the [MULAN repository](http://mulan.sourceforge.net/datasets-mtr.html). |
| `multi-task learning` | the user can load the multi-task learning dataset `dog`, a crowdsourcing dataset first introduced in [Liu et a](https://ieeexplore.ieee.org/document/8440116). More specifically, the dataset contains 800 images of dogs who have been partially labelled by 52 annotators with one of 5 possible breeds. To modify this multi-class problem to a binary problem, we modify the task so that the prediction involves the correct or incorrect labelling by the annotator. In a future version of the software another dataset of the same type will be added.|
| `matrix completion` | the user can load the matrix completion dataset `MovieLens 100K`, a movie rating prediction dataset available by the the [GroupLens lab](https://grouplens.org/datasets/movielens/) that contains 100k ratings from 1000 users on 1700 movies. In a future version of the software larger versions of the movielens dataset will be added  |
| `dyadic prediction` | the user can load dyadic prediction datasets available [here](https://people.montefiore.uliege.be/schrynemackers/datasets). |

## I will upload my own dataset
to be continued...