import streamlit as st
import torch
import GPUtil
import time
import plotly.graph_objects as go


# @st.cache(suppress_st_warning=True, max_entries=2)
def get_available_gpus():
    return GPUtil.getAvailable(
        order="first",
        limit=10,
        maxLoad=100,
        maxMemory=100,
        includeNan=False,
        excludeID=[],
        excludeUUID=[],
    )


def get_gpu_status(deviceIDs):
    gpus_list = []
    for i in deviceIDs:
        gpu = GPUtil.getGPUs()[i]
        # gpus_list.append(str(i)+') '+str(gpu.name) +' / load: '+str(int(gpu.load * 100)) + '% / temp: '+str(int(gpu.temperature))+'C')
        gpus_list.append(str(i) + ") " + str(gpu.name))
    return gpus_list


def get_gpu_status_dict(deviceIDs):
    gpus_list = []
    deviceIDs = get_available_gpus()
    for i in deviceIDs:
        gpu = GPUtil.getGPUs()[i]
        gpus_list.append(
            {
                "id": str(i),
                "name": gpu.name,
                "load": str(int(gpu.load * 100)),
                "temp": str(int(gpu.temperature)),
            }
        )
    return gpus_list


def get_heatmap_fig(dev_id_dict):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=[[i + j for i in range(100)] for j in range(100)]))

    fig.add_trace(
        go.Scatter(
            x=[int(dev_id_dict[i]["load"]) for i in range(len(dev_id_dict))],
            y=[int(dev_id_dict[i]["temp"]) for i in range(len(dev_id_dict))],
            text=["GPU_" + dev_id_dict[i]["id"] for i in range(len(dev_id_dict))],
            mode="markers+text",
            marker=dict(size=[int(20) for i in range(len(dev_id_dict))]),
            textfont=dict(size=20),
        )
    )

    fig.update_layout(
        xaxis_title="load",
        yaxis_title="tempterature C",
    )
    return fig


if "selected_gpu" not in st.session_state:
    st.session_state.selected_gpu = None

if "selected_num_workers" not in st.session_state:
    st.session_state.selected_num_workers = None


if "config" not in st.session_state or st.session_state.config is None:
    st.info(
        "You first have to load a dataset, configure the neural network architecture, select an HPO method and select an logging tool before you can select a GPU"
    )
else:
    # if a gpu is already selected before, then there is not point in displaying the select box and all the other GPUs
    if st.session_state.selected_gpu is None:
        deviceIDs = get_available_gpus()
        if len(deviceIDs) == 0:  # this option should probably default to the cpu
            st.warning("No GPUs detected. Please consider buying one !!! 😅")
        else:
            gpus_dict = get_gpu_status_dict(deviceIDs)
            # for gpu_element in gpus_dict:
            #     col1, col2, col3 = st.columns([2,2,6])
            #     col1.metric(gpu_element['id']+") "+" ".join(gpu_element['name'].split(' ')[1:]), gpu_element['temp']+"°C")
            #     col2.metric('', gpu_element['load']+"%")

            st.plotly_chart(get_heatmap_fig(gpus_dict), use_container_width=True)
            # st.button('Refresh snapshot')
            # selected_gpu = st.radio('Select a GPU:', get_gpu_status(deviceIDs))
            selected_gpu_text = st.selectbox(
                "Select a GPU:", get_gpu_status(deviceIDs) + ["cpu"]
            )

            selected_num_workers = st.number_input(
                "Select number of workers used by the dataloaders",
                min_value=0,
                max_value=10,
                value=4,
                step=1,
            )
            with st.expander("Meaning of the 'num_workers' parameter?"):
                st.markdown(
                    """
                    # Explanation of `num_workers` in PyTorch

                    In PyTorch, particularly when using the `DataLoader` class for loading datasets, the `num_workers` parameter defines the number of subprocesses to use for data loading.

                    - **`num_workers=0`** (default): 
                        - The main process loads the data in the same process. 
                        - Data loading is synchronous and can be slower.
                        
                    - **`num_workers > 0`**: 
                        - Data loading is offloaded to multiple worker processes.
                        - Allows for asynchronous data loading for faster performance.

                    ### Benefits of setting `num_workers` > 0:

                    1. **Speed**: Faster data loading with large datasets or intensive transformations.
                    2. **Utilization**: Utilizes multiple cores of modern CPUs.

                    ### Considerations:

                    1. **Memory**: More workers might consume additional memory.
                    2. **I/O Bottlenecks**: The number of workers might not always linearly speed up data loading due to disk I/O limitations.
                    3. **Thread Safety**: Ensure the dataset access is thread-safe.

                    For best results, experiment with different `num_workers` values to find an optimal balance between speed and resource usage.
                """
                )

            if st.button("Save GPU selection"):
                st.session_state.selected_gpu = selected_gpu_text.split(")")[0]
                st.session_state.selected_num_workers = selected_num_workers
                st.success(st.session_state.selected_gpu + " will be used for training")
    else:
        st.success(
            "GPU_" + st.session_state.selected_gpu + " will be used for training"
        )
        st.success(
            "The dataloaders will use "
            + str(st.session_state.selected_num_workers)
            + " workers"
        )
        # the reset option will reset the selected GPU and will trigger a re-run so that the standard select box is displayed
        if st.button("reset selection"):
            st.session_state.selected_gpu = None
            st.session_state.selected_num_workers = None
            st.experimental_rerun()
