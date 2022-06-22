import streamlit as st
import torch
import GPUtil
import time

# @st.cache(suppress_st_warning=True, max_entries=2)
def get_available_gpus():
    return GPUtil.getAvailable(order = 'first', limit = 10, maxLoad = 1, maxMemory = 1, includeNan=False, excludeID=[], excludeUUID=[])

def get_gpu_status(deviceIDs):
    gpus_list = []
    for i in deviceIDs:
        gpu = GPUtil.getGPUs()[i]
        # gpus_list.append(str(i)+') '+str(gpu.name) +' / load: '+str(int(gpu.load * 100)) + '% / temp: '+str(int(gpu.temperature))+'C')
        gpus_list.append(str(i)+') '+str(gpu.name))
    return gpus_list

def get_gpu_status_dict(deviceIDs):
    gpus_list = []
    deviceIDs = get_available_gpus()
    for i in deviceIDs:
        gpu = GPUtil.getGPUs()[i]
        gpus_list.append({'id': str(i), 'name': gpu.name, 'load': str(int(gpu.load * 100)), 'temp': str(int(gpu.temperature))})
    return gpus_list

if 'selected_gpu' not in st.session_state:
    st.session_state.selected_gpu = None

# if a gpu is already selected before, then there is not point in displaying the select box and all the other GPUs
if st.session_state.selected_gpu is None:
    deviceIDs = get_available_gpus()
    if len(deviceIDs) == 0: # this option should probably default to the cpu
        st.warning('No GPUs detected. Go buy one !!! 😅')
    else:
        for gpu_element in get_gpu_status_dict(deviceIDs):
            col1, col2, col3 = st.columns([2,2,6])
            col1.metric(gpu_element['id']+") "+" ".join(gpu_element['name'].split(' ')[1:]), gpu_element['temp']+"°C")
            col2.metric('', gpu_element['load']+"%")
            col3.write('')
        # st.button('Refresh snapshot')
        # selected_gpu = st.radio('Select a GPU:', get_gpu_status(deviceIDs))
        selected_gpu_text = st.selectbox('Select a GPU:', get_gpu_status(deviceIDs)+['cpu'])

        if st.button('Save GPU selection'):
            st.session_state.selected_gpu = selected_gpu_text.split(')')[0]
            st.success(st.session_state.selected_gpu+' will be used for training')
else:
    st.success(st.session_state.selected_gpu+' will be used for training')
    # the reset option will reset the selected GPU and will trigger a re-run so that the standard select box is displayed
    if st.button('reset selection'):
        st.session_state.selected_gpu = None
        st.experimental_rerun()