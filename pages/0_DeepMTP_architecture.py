import streamlit as st

deepmtp_1 = r'''
    ***
    The baseline architecture of our framework was first popularized by the neural collaborative filtering 
    ([NCF](https://arxiv.org/pdf/1708.05031.pdf)) framework in the field of recommender systems. The architecture 
    successfully approximated standard matrix factorization techniques and showed state-of-the-art performance on
    benchmark datasets. In this work, we show how we can enhance the basic principles of the NCF framework in order to
    build a generalized framework that achieves a competitive performance in all the settings that fall under the 
    umbrella of MTP.
    
    In the proposed architecture shown in the figure below, the network uses two branches to encode the inputs. More
    specifically, the bottom input layer of each branch is comprised of two feature vectors xi and tj , which describe 
    the instance and target of a sample in an MTP problem. Both vectors can be customized to support a range of 
    different MTP formulations. For example, in a typical multi-label classification problem, a one-hot encoded vector 
    will be generated to represent a specific target and used as input to the corresponding branch. Using the same
    principles, in a typical matrix completion problem, we will have to generate one-hot encoded vectors for both 
    instances and targets using their unique ids, very similar to what NCF does.
    
    Above the input layer, we extend the NCF framework by using different types of layers or even entire 
    sub-architectures to better encode the different kinds of inputs the framework may encounter. In cases where no side
    information is provided (for example, the labels in a multi-label classification problem), we use a single 
    fully-connected layer to project the sparse one-hot encoded input vector to a dense embedding. Otherwise, when
    explicit side information is available, we have multiple options, depending on the type of input, from several 
    fully-connected layers (tabular health record data, figure below left) to more specialized architectures based on 
    convolutional neural networks (figure below right) or graph neural networks (hierarchies). The goal of the embedding
    layer in both cases is to project the instances and targets to a lower-dimensional latent space, similarly to what 
    is done with the users and items in the product recommendation problem in [NCF](https://arxiv.org/pdf/1708.05031.pdf)
    
    '''


deepmtp_2 = r'''
    The instance embedding $\mathbf{p_x}$ and target embedding $\mathbf{q_t}$ are then concatenated and passed through 
    a multi-layer neural  network architecture that maps the embeddings to the predicted target value in the following 
    way:
'''

deepmtp_3 = r'''
    where $\mathbf{W}$, $\mathbf{b}$ and $\alpha$ correspond to the weight matrix, bias vector and activation function 
    of the final multi-layer perceptron (MLP) layer. We mainly use the leaky rectified linear unit (Leaky ReLU) as 
    activation function in our framework, but because we also perform experiments with custom architectures from third
    parties instead of the branches, other activation functions may also be utilized (for example, standard ReLU in 
    [Resnet](https://arxiv.org/pdf/1512.03385.pdf)).

    This MLP architecture is able to model more complex, non-linear instance-target relationships compared to a simpler 
    dot product. Even though this idea was popularized by the NCF framework and widely adopted by the CF community, 
    there has been recent work ([paper 1](https://arxiv.org/pdf/1911.07698.pdf), [paper 2](https://arxiv.org/pdf/2005.09683.pdf))
    proposing that the dot product may be highly competitive and cheaper to train. Regardless, we decided that all the 
    experiments shown below should use an MLP and that we will investigate whether the dot product can be a viable
    alternative for the MTP settings in future work.
'''

deepmtp_4 = r'''
    The final output layer consists of a single node that outputs the predicted score $\hat{y}_{\mathbf{xt}}$. In the 
    classification-related MTP settings a sigmoid function is used before the output in order to restrict it to $[0,1]$.
    We facilitate training using different loss functions to accommodate the different categories of MTP problem settings. 
    In classification problems, training is achieved using the binary cross-entropy loss function:
'''

deepmtp_5 = r'''
    On the other hand, in problems that fall into the regression category, we use the squared error loss:
'''

deepmtp_6 = r'''
    In both loss functions, $\mathcal{D}$ denotes the set of known interactions in the training set.
    
    In order to make it more accessible to the reader how training and inference work in our architecture, we make a 
    comparison with a standard neural network in the popular multi-label classification case. The basic neural network 
    will have as many input nodes as instance features and as many output nodes as there are labels (six in the example).
    This means that for the example the neural network will use the pixel values of an image as  input and then output 
    the prediction for every label simultaneously. This procedure is followed during training as well as inference. 
    In our architecture, training and inference are performed in a pairwise manner. Instead of working with all the
    labels of an image simultaneously, we process each instance-target pair separately. Thus, for the same example we 
    detailed earlier, our network will have to input the same image six times to the instance branch and modify the 
    one-hot encoded vector that is passed to the target branch.
    
    It is also important to point out that there are cases in which additional side information is available. 
    These features are usually available for every couple $(\mathbf{x}_i,\mathbf{t}_j)$ in the dataset and have been 
    coined dyadic features in the [literature](https://pubmed.ncbi.nlm.nih.gov/27986855/). Such information requires an extension of our 
    two-branch architecture by a third branch that allows to encode those dyadic features (Figure~\ref{fig:2} right). 
    Similar architectures have been successfully deployed in tensor factorization applications 
    ([paper 1](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-01977-6), [paper 2](https://arxiv.org/pdf/1802.04416.pdf)). 
    In this setting, training and inference remain largerly unchanged, the only difference being the concatenation of 
    three embedding vectors $\mathbf{p_x}$, $\mathbf{q_t}$ and $\mathbf{r_d}$ instead of just two.
    
    Finally, our neural network architecture, combined with the pairwise manner in which we train our models, allows to 
    make predictions for all four validation settings shown in Figure~\ref{fig:5} (Settings A, B, C and D) without 
    having to make modifications in the core training and inference steps. The only stages in the pipeline that need to
    be adapted are the preparation of the dataset splitting as well as the computation of the performance metrics. 

'''


st.title('deepMTP architecture')

st.write(deepmtp_1)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")
with col2:
    st.image('images/convnet_plus_fully_connected.png')
with col3:
    st.write("")

st.write(deepmtp_2)
'''
col4, col5, col6 = st.columns([2, 4, 2])
with col4:
    st.write("")
with col5:
    st.image('images/equation.png')
with col6:
    st.write("")
'''

st.latex(r'''
\mathbf{z_1} = \phi_1(\mathbf{p_x}, \mathbf{q_t})
'''
)

st.latex(r'''
\phi_2(\mathbf{z}_1) = \alpha_{2}(\mathbf{W}_{2}^T \mathbf{z}_1 + \mathbf{b}_2) \\
... \\
'''
)

st.latex(r'''
\phi_L(\mathbf{z}_{L-1}) = \alpha_{L}(\mathbf{W}_{L}^T \mathbf{z}_{L-1} + \mathbf{b}_L)
'''
)

st.latex(r'''
\hat{y}_\mathbf{xt} = \sigma(\mathbf{h}^T \phi_L (\mathbf{z}_{L-1}))
'''
)



# \mathbf{z}_1 & = \phi_1(\mathbf{p_x}, \mathbf{q_t})

st.write(deepmtp_3)
col7, col8, col9 = st.columns([1, 6, 1])
with col7:
    st.write("")
with col8:
    st.image('images/deepMTP_architecture.png')
with col9:
    st.write("")

st.write(deepmtp_4)

st.latex(r'''
{{L}}_{{\mathrm{BCE}}} = -{ \sum_{({\mathbf{x}},{\mathbf{t}}, y) \in \mathcal{D}} {y} \log{\hat{y}_{\mathbf{xt}}} + (1 - y)  \log{(1 - \hat{y}_{\mathbf{xt}}})}
'''
)
st.write(deepmtp_5)
st.latex(r'''
{{L}}_{{\mathrm{MSE}}} = \sum_{({\mathbf{x}},{\mathbf{t}}, y) \in \mathcal{D}} {(y - \hat{y}_{\mathbf{xt}})^2}
''')

st.write(deepmtp_6)
col10, col11, col12 = st.columns([1, 6, 1])
with col10:
    st.write("")
with col11:
    st.image('images/dual_plus_tri-branch_architectures.png')
with col12:
    st.write("")