import os
from multiprocessing import Process
import streamlit as st


class TensorboardSupervisor:
    def __init__(self, log_dir, port):
        self.server = TensorboardServer(log_dir, port)
        self.server.start()
        self.port = str(port)

    def kill(self):
        if self.server.is_alive():
            os.system("kill -9 $(lsof -t -i tcp:" + self.port + ")")
            self.server.terminate()
            self.server.join()


class TensorboardServer(Process):
    def __init__(self, log_dir, port):
        super().__init__()
        self.log_dir = str(log_dir)
        self.port = str(port)
        self.daemon = True

    def run(self):
        os.system("tensorboard --logdir=tensorboard_runs --port=" + str(self.port))


if "config" not in st.session_state or st.session_state.config is None:
    st.info(
        "You first have to load a dataset and configure the neural network architecture"
    )

else:
    hyperopt_methods = ["Hyperband", "Random Search"]
    if "hyperopt_method_index" not in st.session_state:
        st.session_state.hyperopt_method_index = 0
    if "max_budget" not in st.session_state:
        st.session_state.max_budget = 81
    if "eta" not in st.session_state:
        st.session_state.eta = 3
    if "hyperband_selected" not in st.session_state:
        st.session_state.hyperband_selected = False
    if "random_search_budget" not in st.session_state:
        st.session_state.random_search_budget = 1
    if "random_search_selected" not in st.session_state:
        st.session_state.random_search_selected = False

    hyperband_form_submitted = False
    random_search_form_submitted = False

    st.write("## Hyperparameter Optimization")
    st.session_state.hyperopt_method_index = st.selectbox(
        "Hyperparameter Optimization method: ",
        range(len(hyperopt_methods)),
        format_func=lambda x: hyperopt_methods[x],
        index=st.session_state.hyperopt_method_index,
        # on_change=
    )
    hyperopt_method = hyperopt_methods[st.session_state.hyperopt_method_index]

    if hyperopt_method == "Random Search":
        with st.form("random_search_form", clear_on_submit=False):
            st.session_state.random_search_budget = st.number_input(
                "Budget (number of different randomly sampled configurations that will be tested)",
                value=st.session_state.random_search_budget,
            )
            st.session_state.hpo_metric_to_optimize = st.selectbox(
                "metric used to select the best configuration",
                ["loss"] + st.session_state.metrics,
                help='Selecting anything different than "loss" can have performance penalties as all metrics will be calculated for the validation set in every step',
            )
            st.session_state.hpo_metric_average_to_optimize = st.selectbox(
                "metric used to select the best configuration",
                st.session_state.metrics_average,
            )
            random_search_form_submitted = st.form_submit_button(
                "Save Random Search parameters"
            )
        if random_search_form_submitted:
            st.success(
                "Random search parameters saved: (budget: "
                + str(st.session_state.random_search_budget)
                + ")"
            )
            st.session_state.random_search_selected = True
            st.session_state.hyperband_selected = False

    elif hyperopt_method == "Hyperband":
        with st.form("hyperband_form", clear_on_submit=False):
            link = "For a more detailed explanation of how the Hyperband algorithm works click [here](https://share.streamlit.io/diliadis/hyperbandcalculator/main/main.py)"
            st.markdown(link, unsafe_allow_html=True)
            st.session_state.max_budget = st.number_input(
                "Insert a number",
                min_value=1,
                max_value=1000,
                value=int(st.session_state.max_budget),
                step=1,
            )
            st.session_state.eta = st.number_input(
                "Insert a number",
                min_value=1,
                max_value=10,
                value=int(st.session_state.eta),
                step=1,
            )
            st.session_state.hpo_metric_to_optimize = st.selectbox(
                "metric used to select the best configuration",
                ["loss"] + st.session_state.metrics,
                help='Selecting anything different than "loss" can have performance penalties as all metrics will be calculated for the validation set in every step',
            )
            st.session_state.hpo_metric_average_to_optimize = st.selectbox(
                "metric used to select the best configuration",
                st.session_state.metrics_average,
            )
            hyperband_form_submitted = st.form_submit_button(
                "Save Hyperband parameters"
            )
        if hyperband_form_submitted:
            st.success(
                "Hyperband parameters saved: (max_budget: "
                + str(st.session_state.max_budget)
                + ", eta: "
                + str(st.session_state.eta)
                + ")"
            )
            st.session_state.hyperband_selected = True
            st.session_state.random_search_selected = False

    else:
        pass

    if not hyperband_form_submitted and not random_search_form_submitted:
        if st.session_state.hyperband_selected:
            st.success(
                "Hyperband parameters saved: (max_budget: "
                + str(st.session_state.max_budget)
                + ", eta: "
                + str(st.session_state.eta)
                + ")"
            )
        elif st.session_state.random_search_selected:
            st.success(
                "Random search parameters saved: (budget: "
                + str(st.session_state.random_search_budget)
                + ")"
            )
