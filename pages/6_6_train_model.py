import os
from multiprocessing import Process
import streamlit as st
import wandb
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from DeepMTP.main_streamlit import DeepMTP
from DeepMTP.utils.utils import generate_config
from DeepMTP.simple_hyperband_streamlit import BaseWorker
from DeepMTP.simple_hyperband_streamlit import HyperBand
from PIL import Image

from utils import Capturing

from contextlib import redirect_stdout




if 'config' not in st.session_state or st.session_state.config is None:
    st.info('You first have to load a dataset and configure the neural network architecture')

else:    
    st.info('Everything seems to be correctly defined. You can click the button below to start training')           
    streamlit_to_deepMTP_metrics_map = {
        'accuracy': 'accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1 score': 'f1_score',
        'hamming loss': 'hamming_loss',
        'AUROC': 'auroc',
        'AUPR': 'aupr',
        'mean squared error': 'MSE',
        'root mean squared error': 'RMSE',
        'mean absolute error': 'MAE',
        'R2': 'R2',
    }

    # button to start training
    st.session_state.start_experiment_button_pressed = st.button('Start training!!!')

    if st.session_state.start_experiment_button_pressed:

        st.warning('The training process has just started. DO NOT click anything in the application, otherwise all progress will be lost!!!')
        # generate the configuration file
        cond, cond2, cond3, cond4 = None, None, None, None
        cs= CS.ConfigurationSpace()

        # learning rate
        if isinstance(st.session_state.learning_rate, tuple):
            if st.session_state.learning_rate[0] != st.session_state.learning_rate[1]:
                learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=st.session_state.learning_rate[0], upper=st.session_state.learning_rate[1], log=True)
            else:
                learning_rate = CSH.Constant('learning_rate', value=st.session_state.learning_rate[0]) 
        else:
            learning_rate = CSH.Constant('learning_rate', value=st.session_state.learning_rate)

        # dropout rate
        if isinstance(st.session_state.dropout_rate, tuple):
            if st.session_state.dropout_rate[0] != st.session_state.dropout_rate[1]:
                dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=st.session_state.dropout_rate[0], upper=st.session_state.dropout_rate[1], log=False)
            else:
                dropout_rate = CSH.Constant('dropout_rate', value=st.session_state.dropout_rate[0])
        else:
            dropout_rate = CSH.Constant('dropout_rate', value=st.session_state.dropout_rate)

        # batch normalization
        if isinstance(st.session_state.batch_norm, tuple):
            if st.session_state.batch_norm[0] != st.session_state.batch_norm[1]:
                batch_norm = CSH.CategoricalHyperparameter("batch_norm", st.session_state.batch_norm)
            else:
                batch_norm = CSH.Constant('batch_norm', value=st.session_state.batch_norm[0])
        else:
            batch_norm = CSH.Constant('batch_norm', value=st.session_state.batch_norm)
        
        # number of nodes in every layer of the instance branch
        if isinstance(st.session_state.instance_branch_nodes_per_layer, tuple):
            if st.session_state.instance_branch_nodes_per_layer[0] != st.session_state.instance_branch_nodes_per_layer[1]:
                instance_branch_nodes_per_layer = CSH.UniformIntegerHyperparameter('instance_branch_nodes_per_layer', lower=st.session_state.instance_branch_nodes_per_layer[0], upper=st.session_state.instance_branch_nodes_per_layer[1], log=False)
            else:
                instance_branch_nodes_per_layer = CSH.Constant('instance_branch_nodes_per_layer', value=st.session_state.instance_branch_nodes_per_layer[0])
        else:
            instance_branch_nodes_per_layer = CSH.Constant('instance_branch_nodes_per_layer', value=st.session_state.instance_branch_nodes_per_layer)

        # number of layers in the instance branch
        if isinstance(st.session_state.instance_branch_layers, tuple):
            if st.session_state.instance_branch_layers[0] != st.session_state.instance_branch_layers[1]:
                instance_branch_layers = CSH.UniformIntegerHyperparameter('instance_branch_layers', lower=st.session_state.instance_branch_layers[0], upper=st.session_state.instance_branch_layers[1], default_value=1, log=False)
            else:
                instance_branch_layers = CSH.Constant('instance_branch_layers', value=st.session_state.instance_branch_layers[0])
        else:
            instance_branch_layers = CSH.Constant('instance_branch_layers', value=st.session_state.instance_branch_layers)

        # number of nodes in every layer of the target branch
        if isinstance(st.session_state.target_branch_nodes_per_layer, tuple):
            if st.session_state.target_branch_nodes_per_layer[0] != st.session_state.target_branch_nodes_per_layer[1]:
                target_branch_nodes_per_layer = CSH.UniformIntegerHyperparameter('target_branch_nodes_per_layer', lower=st.session_state.target_branch_nodes_per_layer[0], upper=st.session_state.target_branch_nodes_per_layer[1], log=False)
            else:
                target_branch_nodes_per_layer = CSH.Constant('target_branch_nodes_per_layer', value=st.session_state.target_branch_nodes_per_layer[0])
        else:
            target_branch_nodes_per_layer = CSH.Constant('target_branch_nodes_per_layer', value=st.session_state.target_branch_nodes_per_layer)

        # number of layers in the target branch
        if isinstance(st.session_state.target_branch_layers, tuple):
            if st.session_state.target_branch_layers[0] != st.session_state.target_branch_layers[1]:
                target_branch_layers = CSH.UniformIntegerHyperparameter('target_branch_layers', lower=st.session_state.target_branch_layers[0], upper=st.session_state.target_branch_layers[1], default_value=1, log=False)
            else:
                target_branch_layers = CSH.Constant('target_branch_layers', value=st.session_state.target_branch_layers[0])
        else:
            target_branch_layers = CSH.Constant('target_branch_layers', value=st.session_state.target_branch_layers)

        # size of the embedding layer for the instance and target branches
        if isinstance(st.session_state.embedding_size, tuple):
            if st.session_state.embedding_size[0] != st.session_state.embedding_size[1]:
                embedding_size = CSH.UniformIntegerHyperparameter('embedding_size', lower=st.session_state.embedding_size[0], upper=st.session_state.embedding_size[1], log=False)
            else:
                embedding_size = CSH.Constant('embedding_size', value=st.session_state.embedding_size[0])
        else:
            embedding_size = CSH.Constant('embedding_size', value=st.session_state.embedding_size)

        cs.add_hyperparameters([learning_rate, dropout_rate, batch_norm, instance_branch_nodes_per_layer, instance_branch_layers, target_branch_nodes_per_layer, target_branch_layers, embedding_size])
        
        # st.write('learning_rate: '+str(st.session_state.config['learning_rate']))
        # st.write('dropout_rate: '+str(st.session_state.config['dropout_rate']))
        # st.write('batch_norm: '+str(st.session_state.config['batch_norm']))
        # st.write('instance_branch_nodes_per_layer: '+str(st.session_state.config['instance_branch_nodes_per_layer']))
        # st.write('instance_branch_layers: '+str(st.session_state.config['instance_branch_layers']))
        # st.write('target_branch_nodes_per_layer: '+str(st.session_state.config['target_branch_nodes_per_layer']))
        # st.write('target_branch_layers: '+str(st.session_state.config['target_branch_layers']))
        # st.write('embedding_size: '+str(st.session_state.config['embedding_size']))

        if not isinstance(instance_branch_layers, CSH.Constant):
            cond = CS.GreaterThanCondition(dropout_rate, instance_branch_layers, 1)
            cond3 = CS.GreaterThanCondition(batch_norm, instance_branch_layers, 1)
        if not isinstance(target_branch_layers, CSH.Constant):
            cond2 = CS.GreaterThanCondition(dropout_rate, target_branch_layers, 1)
            cond4 = CS.GreaterThanCondition(batch_norm, target_branch_layers, 1)
        if cond and cond2:
            cs.add_condition(CS.OrConjunction(cond, cond2))
        if cond3 and cond4:
            cs.add_condition(CS.OrConjunction(cond3, cond4))
        
        # check if after iterating over the hyperparameters, the user decided to run a singe architecture or we can run Hyperband
        if len(cs.get_hyperparameter_names()) == 0: 
            st.info('Training with a singe configuration.')
            config = generate_config(    
                instance_branch_input_dim = st.session_state.data_info['instance_branch_input_dim'],
                target_branch_input_dim = st.session_state.data_info['target_branch_input_dim'],
                validation_setting = st.session_state.data_info['detected_validation_setting'],
                enable_dot_product_version = True,
                problem_mode = st.session_state.data_info['detected_problem_mode'],
                learning_rate = st.session_state.config['learning_rate'],
                decay = 0,
                batch_norm = st.session_state.config['batch_norm'],
                dropout_rate = st.session_state.config['dropout_rate'],
                momentum = 0.9,
                weighted_loss = False,
                compute_mode = st.session_state['selected_gpu'] if st.session_state['selected_gpu']=='cpu' else 'cuda:'+st.session_state['selected_gpu'],
                train_batchsize = 512,
                val_batchsize = 512,
                num_epochs = st.session_state.config['epochs'],
                num_workers = 8,
                # metrics = ['hamming_loss', 'auroc', 'f1_score', 'aupr', 'accuracy', 'recall', 'precision'],
                # metrics_average = ['macro', 'micro'],
                metrics = [streamlit_to_deepMTP_metrics_map[streamlit_metric] for streamlit_metric in st.session_state.config['metrics']],
                metrics_average = st.session_state.config['metrics_average'],
                patience = 10,

                evaluate_train = True,
                evaluate_val = True,

                verbose = True,
                results_verbose = False,
                use_early_stopping = True,
                use_tensorboard_logger = True,
                wandb_project_name = None if st.session_state['use_wandb'] is None else st.session_state['use_wandb']['project'],
                wandb_project_entity = None if st.session_state['use_wandb'] is None else st.session_state['use_wandb']['entity'],
                metric_to_optimize_early_stopping = 'loss',
                metric_to_optimize_best_epoch_selection = st.session_state.config['metrics'][0],
                instance_branch_architecture = 'MLP',
                use_instance_features = False,
                instance_branch_nodes_reducing_factor = 2,
                instance_branch_nodes_per_layer = st.session_state.config['instance_branch_nodes_per_layer'],
                instance_branch_layers = st.session_state.config['instance_branch_layers'],
                instance_branch_conv_architecture = 'resnet',
                instance_branch_conv_architecture_version = 'resnet101',
                instance_branch_conv_architecture_dense_layers = 1,
                instance_branch_conv_architecture_last_layer_trained = 'last',

                target_branch_architecture = 'MLP',
                use_target_features = True,
                target_branch_nodes_reducing_factor = 2,
                target_branch_nodes_per_layer = st.session_state.config['target_branch_nodes_per_layer'],
                target_branch_layers = st.session_state.config['target_branch_layers'],
                target_branch_conv_architecture = 'resnet',
                target_branch_conv_architecture_version = 'resnet101',
                target_branch_conv_architecture_dense_layers = 1,
                target_branch_conv_architecture_last_layer_trained = 'last',

                embedding_size = st.session_state.config['embedding_size'],

                comb_mlp_nodes_reducing_factor = 2,
                comb_mlp_nodes_per_layer = [2048, 2048, 2048],
                comb_mlp_branch_layers = None, 

                save_model = True,

                eval_every_n_epochs = 10,

                additional_info = {}
            )

            # model = DeepMTP(config)
            # with st.spinner('training...'):
            #     with Capturing() as output:
            #         validation_results = model.train(st.session_state.train, st.session_state.val, st.session_state.test)
            # st.success('Training completed!')
            # st.write(output)

            st.success('Training has just started with the following config: ')
            # st.write(config)
            model = DeepMTP(config)
            with st.spinner('training...'):
                validation_results = model.train(st.session_state.train, st.session_state.val, st.session_state.test)
            st.success('Training completed!')
        
        else: # hyperband will be used

            config = {    
                'hpo_results_path': './hyperband/',

                'instance_branch_input_dim': st.session_state.data_info['instance_branch_input_dim'],
                'target_branch_input_dim': st.session_state.data_info['target_branch_input_dim'],
                'validation_setting': st.session_state.data_info['detected_validation_setting'],
                'enable_dot_product_version': True,
                'problem_mode': st.session_state.data_info['detected_problem_mode'],

                'compute_mode': st.session_state['selected_gpu'] if st.session_state['selected_gpu']=='cpu' else 'cuda:'+st.session_state['selected_gpu'],
                'train_batchsize': 512,
                'val_batchsize': 512,
                'num_epochs': st.session_state.config['epochs'],
                'num_workers': 8,

                'metrics': [streamlit_to_deepMTP_metrics_map[streamlit_metric] for streamlit_metric in st.session_state.config['metrics']],
                'metrics_average': st.session_state.config['metrics_average'],
                'patience': 10,

                'evaluate_train': True,
                'evaluate_val': True,

                'verbose': True,
                'results_verbose': False,
                'use_early_stopping': True,
                'use_tensorboard_logger': True,
                'wandb_project_name': None if st.session_state['use_wandb'] is None else st.session_state['use_wandb']['project'],
                'wandb_project_entity': None if st.session_state['use_wandb'] is None else st.session_state['use_wandb']['entity'],
                'metric_to_optimize_early_stopping': 'loss',
                'metric_to_optimize_best_epoch_selection': 'loss',

                'instance_branch_architecture': 'MLP',

                'target_branch_architecture': 'MLP',

                'save_model': True,

                'eval_every_n_epochs': 10,

                'running_hyperband': True,

                'additional_info': {'eta': st.session_state.eta, 'max_budget': st.session_state.max_budget}
            }

            worker = BaseWorker(
                st.session_state.train, st.session_state.val, st.session_state.test, st.session_state.data_info, config, 'loss'
            )

            hb = HyperBand(
                base_worker=worker,
                configspace=cs,
                eta=st.session_state.eta,
                max_budget=st.session_state.max_budget,
                direction="min",
            )

            hb.run_optimizer()
            st.success('Hyperband completed!')
