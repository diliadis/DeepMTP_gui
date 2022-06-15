import streamlit as st
import wandb

if 'config' not in st.session_state or st.session_state.config is None:
    st.info('You first have to load a dataset and configure the neural network architecture')

else:  

    logging_options = ['Wandb', 'Tensorboard']
    wandb_project_page_link = ''

    if 'logging_option_index' not in st.session_state:
        st.session_state.logging_option_index = 0
    if 'use_wandb' not in st.session_state:
        st.session_state.use_wandb = None

    st.write('---')
    st.session_state.logging_option_index = st.selectbox(
        'Logging_platform: ',
        range(len(logging_options)),
        format_func=lambda x: logging_options[x],
        # index=st.session_state.logging_option_index,
        # on_change=
    )
    logging_option = logging_options[st.session_state.logging_option_index]
    
    if logging_option == 'Tensorboard':
        # st.info('Not supported yet 😳')
        st.write('To start up the tensorboard you just have to open a terminal and navigate to the folder of the current project. After that you can just run the following command:')
        tensorboard_code = '''
        tensorboard --logdir=results
        '''
        st.code(tensorboard_code, language='bash')
        st.write("If everything runs without problems, you will be re-directed to a page in your web browser displaying Tensorboard's user interface")
        # st.write('### TensorBoard options')
        # port = st.number_input('Select a port number in which Tensorboard will be hosted', min_value=0, max_value=65536, value=8503, step=1)
        # (
        #     tensorboard_col1,
        #     tensorboard_col2,
        #     tensorboard_col3,
        #     tensorboard_col4,
        # ) = st.columns(4)
        # local_port_link = 'http://localhost:' + port
        # with tensorboard_col1:
        #     if st.button('Launch Tensorboard'):
        #         if st.session_state['tensorboard_proc'] is None:
        #             st.session_state['tensorboard_proc'] = TensorboardSupervisor(
        #                 '../../runs', port
        #             )
        #             st.info(
        #                 'Tensorboard is now running on port ['
        #                 + port
        #                 + '](%s)' % local_port_link
        #             )
        #             st.session_state['use_tensorboard'] = True
        #         else:
        #             st.info(
        #                 'TensorBoard is already running on port ['
        #                 + port
        #                 + '](%s)' % local_port_link
        #             )
        # with tensorboard_col2:
        #     if st.button('Kill Tensorboard'):
        #         if st.session_state['tensorboard_proc'] is not None:
        #             st.session_state['tensorboard_proc'].kill()
        #             st.session_state['tensorboard_proc'] = None
        #             st.info('Tensorboard is now offline')
        #             st.session_state['use_tensorboard'] = False
        #         else:
        #             st.info('No tensorboard process is currently active.')

    else:
        with st.form(key='wandb_form'):

            st.write('### Weights & Biases')
            wandb_signup_page_link = 'https://wandb.ai/login?signup=true'
            st.write(
                'Weights & Biases (Wandb) is a machine learning platform designed to help with experiment tracking, dataset versioning and model management. If you already have an account with the platform, you can input your api key and entity name, so that all the experiments can be logged to your personal project (You can also create an account [here](%s)). Otherwise, we also provide the option of logging results to the more well known Tensorboard.'
                % wandb_signup_page_link
            )

            # this is just a demo...
            wandb_API_key = st.text_input(
                'API key',
                help='Sets the authentication key associated with your account.',
                type='password',
            )
            wandb_api_key_link = 'Check your API key [here](https://wandb.ai/authorize)'
            st.markdown(wandb_api_key_link, unsafe_allow_html=True)
            wandb_entity = st.text_input(
                'entity',
                help='The entity associated with your run. Usually this is the same as your wandb username',
            )
            wandb_project = st.text_input(
                'project', value='test', help='The project associated with your run.'
            )

            wandb_form_submitted = st.form_submit_button('test connection')
            if wandb_form_submitted:
                try:
                    with st.spinner(
                        'Trying to log a dummy experiment using the supplied info....'
                    ):
                        wandb.login(key=wandb_API_key)
                        wandb_run = wandb.init(
                            entity=wandb_entity, project=wandb_project
                        )
                        wandb_run.finish()

                        # delete the wandb run you just created
                        api=wandb.Api()
                        run = api.run(wandb_entity+'/'+wandb_project+'/'+wandb_run.id)
                        run.delete()

                        st.success(
                            'API key + entity combinations looks valid. Experiment results will be logged to your wandb account.'
                        )
                        wandb_project_page_link = (
                            'https://wandb.ai/' + wandb_entity + '/' + wandb_project
                        )

                        # st.success(
                        #     'Click [here](%s) to access the Wandb project.'
                        #     % wandb_project_page_link
                        # )

                        st.session_state['wandb_proc'] = wandb.init(
                            entity=wandb_entity, project=wandb_project
                        )
                        st.session_state['use_wandb'] = {
                            'entity': wandb_entity,
                            'project': wandb_project,
                            'api_key': wandb_API_key,
                        }
                except ValueError:
                    st.error(
                        'API key must be 40 characters long, yours was '
                        + str(len(wandb_API_key))
                    )
                    st.session_state['use_wandb'] = None
                    pass
                except Exception as e:
                    st.error('API KEY OR ENTITY ARE WRONG. TRY AGAIN')
                    st.session_state['use_wandb'] = None
                    pass

        if st.session_state['use_wandb'] is not None:
            st.success('Click [here](%s) to access the Wandb project.' % wandb_project_page_link)