import streamlit as st
import time
import pandas as pd
import numpy as np

values = ['val_1', 'val_2']

if 'example_index' not in st.session_state:
    st.session_state.example_index = 0

st.write('## Hyperparameter Optimization')
val = st.selectbox(
    'Select one value: ',
    range(len(values)),
    format_func=lambda x: values[x],
    index=st.session_state.example_index,
)
st.write(str(val)+' '+str(time.time()))
st.session_state.example_index = val

st.write('hello', 'there')
st.write('hello')

col1, col2 = st.columns([2,8])

for i in range(10):

    if i == 5:
        st.write(20*'=')
    else:
        col1, col2 = st.columns([2,8])

        col1.write('Validation: '+str(i)+'... ')

        col2.write('Done')

run_story_header = ['hello', '0', 0.222, 'train']
train_progress_df = pd.DataFrame(np.array(['' for h_idx in range(len(run_story_header))]).reshape(1,-1), columns=run_story_header)
train_progress_table = st.table()
train_progress_table.add_rows(train_progress_df)

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])


st.line_chart()

