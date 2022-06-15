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