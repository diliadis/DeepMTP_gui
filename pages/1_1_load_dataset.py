
from time import sleep
import time
import streamlit as st
import pandas as pd
import csv
from DeepMTP.dataset import load_process_MLC, load_process_MTR, load_process_DP, process_dummy_MLC, process_dummy_MTR, process_dummy_DP, load_process_MC, load_process_MTL
from DeepMTP.utils.data_utils import data_process
from utils import Capturing, get_mtp_settings_info

def reset_dataset_option_index():
    st.session_state.dataset_option_index = 0

@st.experimental_memo(max_entries=2)
def load_dataset(mtp_problem_setting_name, dataset_name):
    if mtp_problem_setting_name == 'multi-label classification':
        data = load_process_MLC(dataset_name=dataset_name, print_mode='dev')
    elif mtp_problem_setting_name == 'multivariate regression':
        data = load_process_MTR(dataset_name=dataset_name, print_mode='dev')
    elif mtp_problem_setting_name == 'multi-task learning':
        data = load_process_MTL(dataset_name=dataset_name, print_mode='dev')
    elif mtp_problem_setting_name == 'matrix completion':
        data = load_process_MC(dataset_name=dataset_name, print_mode='dev')
    elif mtp_problem_setting_name == 'dyadic prediction':
        data = load_process_DP(dataset_name=dataset_name, print_mode='dev')
    else:
        st.error('invalid MTP problem setting selected')

    return data

info_per_mtp_setting_per_dataset = get_mtp_settings_info()

dataset_mode_options = ['Use a build-in dataset', 'I will upload my own dataset']

mlc_dataset_names = ['Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500', 'yeast']
mtr_dataset_names = ['atp1d', 'atp7d', 'oes97', 'oes10', 'rf1', 'rf2', 'scm1d', 'scm20d', 'edm', 'sf1', 'sf2', 'jura', 'wq', 'enb', 'slump', 'andro', 'osales', 'scpf']
mtl_dataset_names = ['dog']
mc_dataset_names = ['ml-100k']
dp_dataset_names = ['srn', 'ern', 'dpie', 'dpii']

datasets_per_mtp_problem_setting = {
    'None': [],
    'multi-label classification': mlc_dataset_names,
    'multivariate regression': mtr_dataset_names,
    'multi-task learning': mtl_dataset_names,
    'matrix completion': mc_dataset_names,
    'dyadic prediction': dp_dataset_names
}
mtp_problem_setting_names = list(datasets_per_mtp_problem_setting.keys())

# initializing session_state variables
if 'built_in_dataset_loaded' not in st.session_state:
    st.session_state.built_in_dataset_loaded = False
if 'custom_dataset_loaded' not in st.session_state:
    st.session_state.custom_dataset_loaded = False    
if 'dataset_mode_option_index' not in st.session_state:
    st.session_state.dataset_mode_option_index = 0
if 'mtp_problem_setting_option_index' not in st.session_state:
    st.session_state.mtp_problem_setting_option_index = 0
if 'dataset_option_index' not in st.session_state:
    st.session_state.dataset_option_index = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'train' not in st.session_state:
    st.session_state.train = None
if 'val' not in st.session_state:
    st.session_state.val = None
if 'test' not in st.session_state:
    st.session_state.test = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None

if 'y_train_file' not in st.session_state:
    st.session_state.y_train_file = None
if 'y_val_file' not in st.session_state:
    st.session_state.y_val_file = None
if 'y_test_file' not in st.session_state:
    st.session_state.y_test_file = None
if 'X_train_instance_file' not in st.session_state:
    st.session_state.X_train_instance_file = None
if 'X_val_instance_file' not in st.session_state:
    st.session_state.X_val_instance_file = None
if 'X_test_instance_file' not in st.session_state:
    st.session_state.X_test_instance_file = None
if 'X_train_target_file' not in st.session_state:
    st.session_state.X_train_target_file = None
if 'X_val_target_file' not in st.session_state:
    st.session_state.X_val_target_file = None
if 'X_test_target_file' not in st.session_state:
    st.session_state.X_test_target_file = None

if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_val' not in st.session_state:
    st.session_state.y_val = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'X_train_instance' not in st.session_state:
    st.session_state.X_train_instance = None
if 'X_val_instance' not in st.session_state:
    st.session_state.X_val_instance = None
if 'X_test_instance' not in st.session_state:
    st.session_state.X_test_instance = None
if 'X_train_target' not in st.session_state:
    st.session_state.X_train_target = None
if 'X_val_target' not in st.session_state:
    st.session_state.X_val_target = None
if 'X_test_target' not in st.session_state:
    st.session_state.X_test_target = None

if 'y_train_has_header' not in st.session_state:
    st.session_state.y_train_has_header = None
if 'y_val_has_header' not in st.session_state:
    st.session_state.y_val_has_header = None
if 'y_test_has_header' not in st.session_state:
    st.session_state.y_test_has_header = None
if 'X_train_instance_has_header' not in st.session_state:
    st.session_state.X_train_instance_has_header = None
if 'X_val_instance_has_header' not in st.session_state:
    st.session_state.X_val_instance_has_header = None
if 'X_test_instance_has_header' not in st.session_state:
    st.session_state.X_test_instance_has_header = None
if 'X_train_target_has_header' not in st.session_state:
    st.session_state.X_train_target_has_header = None
if 'X_val_target_has_header' not in st.session_state:
    st.session_state.X_val_target_has_header = None
if 'X_test_target_has_header' not in st.session_state:
    st.session_state.X_test_target_has_header = None

st.session_state.dataset_mode_option_index = st.radio(
    'Upload a dataset:',
    range(len(dataset_mode_options)),
    format_func=lambda x: dataset_mode_options[x],
    # index=st.session_state.dataset_mode_option_index,
    horizontal=True
)
dataset_mode_option = dataset_mode_options[st.session_state.dataset_mode_option_index]

st.write('---')

if dataset_mode_option == 'Use a build-in dataset':
    # select box for different MTP problem settings. After selecting one of the MTP settings, the select box of the datasets is reset
    st.session_state.mtp_problem_setting_option_index = st.selectbox(
        'Select an MTP problem setting: ',
        range(len(datasets_per_mtp_problem_setting)),
        format_func=lambda x: mtp_problem_setting_names[x],
        index=st.session_state.mtp_problem_setting_option_index,
        on_change=reset_dataset_option_index
    )

    if st.session_state.mtp_problem_setting_option_index !=0:
        # select box with the available datasets for a given MTP problem setting
        st.session_state.dataset_option_index = st.selectbox(
            'Select one of the '+mtp_problem_setting_names[st.session_state.mtp_problem_setting_option_index]+' datasets:',
            range(len(datasets_per_mtp_problem_setting[mtp_problem_setting_names[st.session_state.mtp_problem_setting_option_index]])),
            format_func=lambda x: datasets_per_mtp_problem_setting[mtp_problem_setting_names[st.session_state.mtp_problem_setting_option_index]][x],
            index=st.session_state.dataset_option_index
        )

        # translating the selected ids to the names of the MTP problem setting and the dataset
        selected_mtp_problem_setting_name = mtp_problem_setting_names[st.session_state.mtp_problem_setting_option_index]
        selected_dataset_name = datasets_per_mtp_problem_setting[selected_mtp_problem_setting_name][st.session_state.dataset_option_index]
        if selected_mtp_problem_setting_name in info_per_mtp_setting_per_dataset.keys():
            st.dataframe(info_per_mtp_setting_per_dataset[selected_mtp_problem_setting_name][info_per_mtp_setting_per_dataset[selected_mtp_problem_setting_name]['name'] == selected_dataset_name])


        if st.button('Load dataset'):
            # loading the dataset
            with st.spinner('Loading...'):
                st.info('Loading '+selected_dataset_name+' ('+selected_mtp_problem_setting_name+')')
                with Capturing() as output:
                    st.session_state.data = load_dataset(selected_mtp_problem_setting_name, selected_dataset_name)
                for out in output:
                    if out.startswith('info:'):
                        st.info(out[len('info:'):])
                    elif out.startswith('error:'):
                        st.error(out[len('error:'):])
                    elif out.startswith('warning:'):
                        st.warning(out[len('warning:'):])
                st.session_state.built_in_dataset_loaded = True
                st.success('Done')

else:

    with st.form("my_form"):
        st.header('Interaction data')
        st.session_state.y_train_has_header = st.radio('train interaction file has header', [True, False], horizontal=True)
        st.session_state.y_train_file = st.file_uploader('Upload the train interaction file')
        st.session_state.y_val_has_header = st.radio('val interaction file has header', [True, False], horizontal=True)
        st.session_state.y_val_file = st.file_uploader('Upload the val interaction file')
        st.session_state.y_test_has_header = st.radio('test interaction file has header', [True, False], horizontal=True)
        st.session_state.y_test_file = st.file_uploader('Upload the test interaction file')
        st.write('---')
        st.header('Instance features')
        st.session_state.X_train_instance_has_header = st.radio('train instance features file has header', [True, False], horizontal=True)
        st.session_state.X_train_instance_file = st.file_uploader('Upload the train instance features file')
        st.session_state.X_val_instance_has_header = st.radio('val instance features file has header', [True, False], horizontal=True)
        st.session_state.X_val_instance_file = st.file_uploader('Upload the val instance features file')
        st.session_state.X_test_instance_has_header = st.radio('test instance features file has header', [True, False], horizontal=True)
        st.session_state.X_test_instance_file = st.file_uploader('Upload the test instance features file')
        st.write('---')
        st.header('Target features')
        st.session_state.X_train_target_has_header = st.radio('train target features file has header', [True, False], horizontal=True)
        st.session_state.X_train_target_file = st.file_uploader('Upload the train target features file')
        st.session_state.X_val_target_has_header = st.radio('val target features file has header', [True, False], horizontal=True)
        st.session_state.X_val_target_file = st.file_uploader('Upload the val target features file')
        st.session_state.X_test_target_has_header = st.radio('test target features file has header', [True, False], horizontal=True)
        st.session_state.X_test_target_file = st.file_uploader('Upload the test target features file')

        # Every form must have a submit button.
        submitted = st.form_submit_button('Load datasets')

    if submitted:
        if st.session_state.y_train_file is None:
            st.error('You have to at least upload the interaction data for the training set')
        else:
            with st.spinner('Loading files...'):
                if st.session_state.y_train_file is not None: 
                    st.session_state.y_train = pd.read_csv(st.session_state.y_train_file, header=st.session_state.y_train_has_header if st.session_state.y_train_has_header else None)
                    if not st.session_state.y_train_has_header:
                        st.session_state.y_train = st.session_state.y_train.to_numpy()
                if st.session_state.y_val_file is not None: 
                    st.session_state.y_val = pd.read_csv(st.session_state.y_val_file, header=st.session_state.y_val_has_header if st.session_state.y_val_has_header else None)
                    if not st.session_state.y_val_has_header:
                        st.session_state.y_val = st.session_state.y_val.to_numpy()
                if st.session_state.y_test_file is not None: 
                    st.session_state.y_test = pd.read_csv(st.session_state.y_test_file, header=st.session_state.y_test_has_header if st.session_state.y_test_has_header else None)
                    if not st.session_state.y_test_has_header:
                        st.session_state.y_test = st.session_state.y_test.to_numpy()

                if st.session_state.X_train_instance_file is not None: 
                    st.session_state.X_train_instance = pd.read_csv(st.session_state.X_train_instance_file, header=st.session_state.X_train_instance_has_header if st.session_state.X_train_instance_has_header else None)
                    if not st.session_state.X_train_instance_has_header:
                        st.session_state.X_train_instance = st.session_state.X_train_instance.to_numpy()
                if st.session_state.X_val_instance_file is not None: 
                    st.session_state.X_val_instance = pd.read_csv(st.session_state.X_val_instance_file, header=st.session_state.X_val_instance_has_header if st.session_state.X_val_instance_has_header else None)
                    if not st.session_state.X_val_instance_has_header:
                        st.session_state.X_val_instance = st.session_state.X_val_instance.to_numpy()
                if st.session_state.X_test_instance_file is not None: 
                    st.session_state.X_test_instance = pd.read_csv(st.session_state.X_test_instance_file, header=st.session_state.X_test_instance_has_header if st.session_state.X_test_instance_has_header else None)
                    if not st.session_state.X_test_instance_has_header:
                        st.session_state.X_test_instance = st.session_state.X_test_instance.to_numpy()       

                if st.session_state.X_train_target_file is not None: 
                    st.session_state.X_train_target = pd.read_csv(st.session_state.X_train_target_file, header=st.session_state.X_train_target_has_header if st.session_state.X_train_target_has_header else None)
                    if not st.session_state.X_train_target_has_header:
                        st.session_state.X_train_target = st.session_state.X_train_target.to_numpy()
                if st.session_state.X_val_target_file is not None: 
                    st.session_state.X_val_target = pd.read_csv(st.session_state.X_val_target_file, header=st.session_state.X_val_target_has_header if st.session_state.X_val_target_has_header else None)
                    if not st.session_state.X_val_target_has_header:
                        st.session_state.X_val_target = st.session_state.X_val_target.to_numpy()
                if st.session_state.X_test_target_file is not None: 
                    st.session_state.X_test_target = pd.read_csv(st.session_state.X_test_target_file, header=st.session_state.X_test_target_has_header if st.session_state.X_test_target_has_header else None)
                    if not st.session_state.X_test_target_has_header:
                        st.session_state.X_test_target = st.session_state.X_test_target.to_numpy()
                            
            st.success('Done')


    if st.session_state.y_train is not None:
        st.write('---')
        st.header('Interaction data')
        st.subheader('Train interaction data')
        st.dataframe(st.session_state.y_train)
        st.subheader('Val interaction data')
        st.dataframe(st.session_state.y_val)
        st.subheader('Test interaction data')
        st.dataframe(st.session_state.y_test)

        st.write('---')
        st.header('Instance features')
        st.subheader('Train instance features')
        st.dataframe(st.session_state.X_train_instance)
        st.subheader('Val instance features')
        st.dataframe(st.session_state.X_val_instance)
        st.subheader('Test instance features')
        st.dataframe(st.session_state.X_test_instance)

        st.write('---')
        st.header('Target features')
        st.subheader('Train target features')
        st.dataframe(st.session_state.X_train_target)
        st.subheader('Val target features')
        st.dataframe(st.session_state.X_val_target)
        st.subheader('Test target features')
        st.dataframe(st.session_state.X_test_target)

        st.session_state.data = {
            'train': {
                'y': st.session_state.y_train,
                'X_instance': st.session_state.X_train_instance, 
                'X_target': st.session_state.X_train_target
            }, 
            'test': {
                'y': st.session_state.y_test, 
                'X_instance': st.session_state.X_test_instance,
                'X_target': st.session_state.X_test_target
            }, 
            'val': {
                'y': st.session_state.y_val, 
                'X_instance': st.session_state.X_val_instance, 
                'X_target': st.session_state.X_val_target
            }
        }
        st.session_state.custom_dataset_loaded = True

if (dataset_mode_option == 'Use a build-in dataset' and st.session_state.built_in_dataset_loaded == True) or (dataset_mode_option == 'I will upload my own dataset' and st.session_state.custom_dataset_loaded == True):
    if st.session_state.data is not None:
        if st.session_state.train is None and st.session_state.val is None and st.session_state.test is None:
            st.header('Preprocess data')
            with st.spinner('Processing...'):
                with Capturing() as output:
                    st.session_state.train, st.session_state.val, st.session_state.test, st.session_state.data_info = data_process(st.session_state.data, validation_setting='B', verbose=True, print_mode='dev')
                for out in output:
                    if out.startswith('info:'):
                        if 'Passed' in out:
                            st.success(out[len('info:'):])
                        else:
                            st.info(out[len('info:'):])
                    elif out.startswith('error:'):
                        st.error(out[len('error:'):])
                    elif out.startswith('warning:'):
                        st.warning(out[len('warning:'):])
            st.success('Done')
            st.write('---')
            st.info('You can now head to the next page in order to configure the architecture of the neural network.')
        else:
            st.info('A dataset is alread loaded. You can now head to the next page in order to configure the architecture of the neural network.')

        if st.button('Reset dataset selection'):
            st.session_state.built_in_dataset_loaded = False
            st.session_state.custom_dataset_loaded = False    
            st.session_state.dataset_mode_option_index = 0
            st.session_state.mtp_problem_setting_option_index = 0
            st.session_state.dataset_option_index = 0
            st.session_state.data = None
            st.session_state.train = None
            st.session_state.val = None
            st.session_state.test = None
            st.session_state.data_info = None

            st.session_state.y_train_file = None
            st.session_state.y_val_file = None
            st.session_state.y_test_file = None
            st.session_state.X_train_instance_file = None
            st.session_state.X_val_instance_file = None
            st.session_state.X_test_instance_file = None
            st.session_state.X_train_target_file = None
            st.session_state.X_val_target_file = None
            st.session_state.X_test_target_file = None

            st.session_state.y_train = None
            st.session_state.y_val = None
            st.session_state.y_test = None
            st.session_state.X_train_instance = None
            st.session_state.X_val_instance = None
            st.session_state.X_test_instance = None
            st.session_state.X_train_target = None
            st.session_state.X_val_target = None
            st.session_state.X_test_target = None
            st.experimental_rerun()

