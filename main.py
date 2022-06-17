import streamlit as st
from PIL import Image
from utils import clearConsole
# streamlit run main_app.py --server.port 8502

def MTP_section():
    MTP_intro = r"""
    ---
    **What is MTP??**
    
    Multi-target prediction (MTP) serves as an umbrella term for machine learning tasks that concern the simultaneous 
    prediction of multiple target variables. These include:
    
    - Multi-label Classification
    - Multivariate Regression
    - Multitask Learning
    - Hierarchical Multi-label Classification
    - Dyadic Prediction
    - Zero-shot Learning
    - Matrix Completion
    - (Hybrid) Matrix Completion
    - Cold-start Collaborative Filtering
    
    Despite the significant similarities, all these domains have evolved separately into distinct research areas over the 
    last two decades.
    
    In this work we present a generic deep learning methodology that can be used for a wide range of
    multi-target prediction problems (**deepMTP**).
    
    ---
    
    **Figuring out the most appropriate MTP setting can be tricky!!!**
    
    The selection of the most appropriate MTP problem setting for a given dataset can be a challenge mainly because of the 
    small details that distinguish them. We were able to distil our experience in the area and create a custom questionnaire
    that can help users to make the correct decision. The current version of the questionnaire contains the following
    questions:
    
    - **Q1**: Is it expected to encounter novel instances during testing? **(yes/no)**
    - **Q2**: Is it expected to encounter novel targets during testing? **(yes/no)**
    - **Q3**: Is there side information available for the instances? **(yes/no)**
    - **Q4**: Is there side information available for the targets? **(yes/no)**
    - **Q5**: Is the score matrix fully observed? **(yes/no)**
    - **Q6**: What is the type of the target variable? **(binary/nominal/ordinal/real-valued)**
    
    The questionnaire is partly answered automatically with our framework from the characteristics of the dataset. 
    There are also questions that currently can only be answered by the user and that have been carefully designed to 
    extract his/her intentions about the given problem. We imagine that by using a graphical interface that accepts the test
    set, a future version can automatically detect whether the user expects a generalization to unseen instances or targets.
    
    These questions are designed to determine the possibility of encountering novel instances or targets during the test 
    phase, the availability of usable side information in the form of relations or representations for instances and 
    targets, the sparsity of the score matrix and the type of values inside the matrix. The aforementioned questions 
    generate **128 different combinations**. We have internally annotated the most popular cases with the appropriate 
    multi-target prediction setting (see Table below), thus transferring our expert knowledge into the rule-based system.
    
    #
    
    | Q1  | Q2 | Q3  | Q4 | Q5  | Q6 | MTP method |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | yes  | no  | yes | no  | yes | binary  | **Multi-label classification**  |
    | yes  | no  | yes | no  | yes | real-valued  | **Multivariate regression** |
    | yes  | no  | yes | no  | no  | -  | **Multi-task learning** |
    | yes  | no  | yes | yes (hierarchy)  | yes | binary | **Hierarchical Multi-label classification** |
    | yes  | no  | yes | yes | no  | - | **Dyadic prediction** |
    | yes  | yes | yes | yes | no  | - | **Zero-shot learning** |
    | no   | no  | no  | no  | no  | - | **Matrix completion** |
    | no   | no  | yes | yes | no  | - | **Hybrid Matrix completion** |
    | yes  | yes | yes | yes | no  | - | **Cold-start Collaborative filtering** |
    
    #
    
    There are, however, some specific combinations of characteristics that make the resulting example unable to be annotated.
    These examples usually try to generalize to novel instances or targets without providing the appropriate side 
    information.
    The mentioned differences in the availability of side information that is traditionally associated with each MTP problem 
    setting has led to the distinction of several validation settings. In order to support the different inference cases
    of all the MTP problem settings, we define the following four experimental settings under which one can make predictions
    for new couples $(\mathbf{x}_i,\mathbf{t}_j)$. 
    
    - Setting A: Both $\mathbf{x}_i$ and $\mathbf{t}_j$ are observed during training.
    - Setting B: All targets $\mathbf{t}_j$ are observed during training and the goal is to make predictions for unseen instances
    - Setting C: All instances $\mathbf{x}_i$ are observed during training and the goal is to make predictions for unseen targets $\mathbf{t}_j$.
    - Setting D: Neither $\mathbf{x}_i$ nor $\mathbf{t}_j$ is observed during training.
    
    #
    """
    clearConsole()
    st.write(MTP_intro)

    val_settings_image = Image.open("images/validation_settings_transparent.png")
    st.image(
        val_settings_image,
        caption="The four validation settings in MTP",
        use_column_width=True,
    )

# reseting various variables that determine which components are showing in the other pages of the app
if 'start_experiment_button_pressed' not in st.session_state:
    st.session_state.start_experiment_button_pressed = False
st.session_state.start_experiment_button_pressed = False

st.set_page_config(
    page_title='DeepMTP', 
    layout='centered',
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }    
)

st.sidebar.image('images/logo_transparent.png', use_column_width=True)

st.title('DeepMTP (experimental)')

MTP_section()
