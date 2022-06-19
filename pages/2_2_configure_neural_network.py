import streamlit as st

def get_range(minmax_val, full_range):    
    return [minmax_val[0]] if minmax_val[0] == minmax_val[1] else full_range[full_range.index(minmax_val[0]):full_range.index(minmax_val[1])+1]

if 'data' not in st.session_state or st.session_state.data is None:
    st.info('You first have to load a dataset in the previous page')
else:
    learning_rate_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    dropout_rate_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    batch_norm_range = ['True', 'False']
    instance_branch_nodes_per_layer_range = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    instance_branch_layers_range = [1, 2, 3, 4, 5]
    target_branch_nodes_per_layer_range = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    target_branch_layers_range = [1, 2, 3, 4, 5]
    embedding_size_range = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    number_of_layers_per_final_mlp_range = [1, 2, 3, 4, 5]

    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = None
    if 'learning_rate' not in st.session_state:
        st.session_state.learning_rate = None
    if 'epochs' not in st.session_state:
        st.session_state.epochs = None
    if 'dropout_rate' not in st.session_state:
        st.session_state.dropout_rate = None
    if 'batch_norm' not in st.session_state:
        st.session_state.batch_norm = None
    if 'instance_branch_nodes_per_layer' not in st.session_state:
        st.session_state.instance_branch_nodes_per_layer = None
    if 'instance_branch_layers' not in st.session_state:
        st.session_state.instance_branch_layers = None
    if 'target_branch_nodes_per_layer' not in st.session_state:
        st.session_state.target_branch_nodes_per_layer = None
    if 'target_branch_layers' not in st.session_state:
        st.session_state.target_branch_layers = None
    if 'embedding_size' not in st.session_state:
        st.session_state.embedding_size = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'averaging' not in st.session_state:
        st.session_state.metrics_average = None

    if 'config' not in st.session_state:
        st.session_state.config = None

    if st.session_state.config is None:
        with st.form('architecture_form', clear_on_submit=False):

            st.write('### Training parameters')
            st.session_state.batch_size = st.select_slider(
                'batch size',
                options=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                value=256,
            )
            st.session_state.learning_rate = st.select_slider(
                'learning rate',
                options=learning_rate_range,
                value=(0.0001, 0.001),
            )
            st.session_state.epochs = st.slider(
                'Number of epochs',
                min_value=20,
                max_value=500,
                value=100,
                step=1,
            )

            st.write('***')
            st.write('### Regularization')
            st.session_state.dropout_rate = st.select_slider(
                'dropout_rate', 
                options=dropout_rate_range, 
                value=(0, 0.3), 
            )
            st.session_state.batch_norm = st.select_slider(
                'branch normalization', 
                options=batch_norm_range, 
                value=('True', 'False') 
            )

            st.write('***')
            st.write('### Architecture characteristics')
            st.session_state.instance_branch_nodes_per_layer = st.select_slider(
                'Size of the input layer for the instance branch',
                options=instance_branch_nodes_per_layer_range,
                value=[8, 2048],
            )
            with st.expander('see hint'):
                st.write(
                    'You define the number of input nodes for the instance branch (Choose a number based on the number of instance features you have)'
                )
                st.image('images/architecture_parameters_hints/instace_branch_input_size.png')

            st.write('***')
            st.session_state.instance_branch_layers = st.select_slider(
                'Number of layers for the instance branch?',
                options=instance_branch_layers_range,
                value=(1, 2),
            )
            with st.expander('see hint'):
                st.write('You define the number of layers in the instance branch')
                st.image('images/architecture_parameters_hints/instance_branch_depth.png')

            st.write('***')
            st.session_state.target_branch_nodes_per_layer = st.select_slider(
                'Size of the input layer for the target branch',
                options=target_branch_nodes_per_layer_range,
                value=(8, 2048),
            )
            with st.expander('see hint'):
                st.write(
                    'You define the number of input nodes for the target branch (Choose a number based on the number of target features you have)'
                )
                st.image('images/architecture_parameters_hints/target_branch_input_size_v2.png')

            st.write('***')
            st.session_state.target_branch_layers = st.select_slider(
                'Number of layers for the target branch?',
                options=target_branch_layers_range,
                value=[1, 1],
            )
            with st.expander('see hint'):
                st.write('You define the number of layers in the target branch')
                st.image('images/architecture_parameters_hints/target_branch_depth.png')

            st.write('***')
            st.session_state.embedding_size = st.select_slider(
                'Size of the embedding layer',
                options=embedding_size_range,
                value=(32, 32),
            )
            with st.expander('see hint'):
                st.write('You define the number of input nodes for the embedding layer')
                st.image('images/architecture_parameters_hints/embedding_layer_size.png')
            
            # st.write('***')
            # number_of_layers_per_final_mlp = st.select_slider('Number of layers for the final branch?', options=number_of_layers_per_final_mlp_range, value=(1,2), key='number_of_layers_per_final_mlp')
            # with st.expander('see hint'):
            #     st.write('You define the number of layers in the final branch')
            #     st.image('images/architecture_parameters_hints/final_mlp_depth.png')

            st.write('***')
            st.write('## Performance Metrics')
            metrics_options = None
            if st.session_state.data_info['detected_problem_mode'] == 'classification':
                metrics_options = [
                    'accuracy',
                    'recall',
                    'precision',
                    'f1 score',
                    'hamming loss',
                    'AUROC',
                    'AUPR',
                ]
            else:
                metrics_options = [
                    'mean squared error',
                    'root mean squared error',
                    'mean absolute error',
                    'R2',
                ]

            st.session_state.metrics = st.multiselect(
                'Metrics to calculate',
                metrics_options,
                default=metrics_options[0:2],
                help='Different metrics are offered based on the task (Classification, Regression, Ranking...)',
            )
            if not st.session_state.metrics:
                st.info(
                    'No metrics are selected. Using '
                    + str(metrics_options[0:2])
                    + ' as the default'
                )
                metrics = metrics_options[0:2]

            st.session_state.metrics_average = st.multiselect(
                'Averaging ',
                ['instance-wise', 'macro', 'micro'],
                default=['macro', 'micro'],
                help='different averaging methods are offered for different validation settings',
            )
            if not st.session_state.metrics_average:
                st.info('No averaging method is selected. Using micro-averaging as the default')
                st.session_state.metrics_average.append('micro')

            with st.expander('see hint'):
                st.write(
                    'The calculation of performance measures for all labels can be achieved using three averaging operations, called macro-averaging, micro-averaging, and instance-wise-averaging.'
                )
                st.image('images/averaging.jpg')

            submitted = st.form_submit_button('Save architecture choices')

        if submitted:
            # st.write('batch_size: '+str(st.session_state.batch_size))
            # st.write('learning_rate: '+str(st.session_state.learning_rate))
            # st.write('epochs: '+str(st.session_state.epochs))
            # st.write('dropout_rate: '+str(st.session_state.dropout_rate))
            # st.write('batch_norm: '+str(st.session_state.batch_norm))
            # st.write('instance_branch_nodes_per_layer: '+str(st.session_state.instance_branch_nodes_per_layer))
            # st.write('instance_branch_layers: '+str(st.session_state.instance_branch_layers))
            # st.write('target_branch_nodes_per_layer: '+str(st.session_state.target_branch_nodes_per_layer))
            # st.write('target_branch_layers: '+str(st.session_state.target_branch_layers))
            # st.write('embedding_size: '+str(st.session_state.embedding_size))
            # st.write('metrics: '+str(st.session_state.metrics))
            # st.write('metrics_average: '+str(st.session_state.metrics_average))

            st.session_state.config = {
                'batch_size': st.session_state.batch_size,
                'learning_rate': st.session_state.learning_rate,
                'epochs': st.session_state.epochs,
                'dropout_rate': st.session_state.dropout_rate,
                'batch_norm': st.session_state.batch_norm,
                'instance_branch_nodes_per_layer': st.session_state.instance_branch_nodes_per_layer,
                'instance_branch_layers': st.session_state.instance_branch_layers,
                'target_branch_nodes_per_layer': st.session_state.target_branch_nodes_per_layer,
                'target_branch_layers': st.session_state.target_branch_layers,
                'embedding_size': st.session_state.embedding_size,
                'metrics': st.session_state.metrics,
                'metrics_average': st.session_state.metrics_average
            }
            st.success('Configuration saved ✅')

    st.header('Selected Hyperparameters')
    if st.session_state.config is None:
        st.info('Select the ranges for the hyperparameters above and click "Save architecture choices" to save them.')
    else:
        st.write(st.session_state.config)

        if st.button('Reset hyperparameters'):
            st.session_state.config = None