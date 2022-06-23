from io import StringIO 
import sys
import os
import streamlit as st
import pandas as pd

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def clearConsole():
    command = "clear"
    if os.name in ("nt", "dos"):  # If Machine is running on Windows, use cls
        command = "cls"
    os.system(command)

@st.experimental_memo(max_entries=2)
def get_mtp_settings_info():
    info_per_mtp_setting = {
    'multi-label classification': pd.DataFrame([
        ['Corel5k', 5000, 499, 374, None],
        ['bibtex', 7395, 1836, 159, None],
        ['birds', 645, 260, 19, None],
        ['delicious', 16105, 500, 983, None], 
        ['emotions', 593, 72, 6, None],
        ['enron', 1702, 1001, 53, None],
        ['genbase', 662, 1186, 27, None],
        ['mediamill', 43907, 120, 101, None],
        ['medical', 978, 1449, 45, None],
        ['rcv1subset1', 6000, 47236, 101, None],
        ['rcv1subset2', 6000, 47236, 101, None],
        ['rcv1subset3', 6000, 47236, 101, None],
        ['rcv1subset4', 6000, 47229, 101, None],
        ['rcv1subset5', 6000, 47235, 101, None],
        ['scene', 2407, 294, 6, None],
        ['tmc2007_500', 28596, 500, 22, None],
        ['yeast', 2417, 103, 14, None],
    ], columns=['name', '#instance', '#instance_features', '#targets', '#target_features'])
    ,
    'multivariate regression': pd.DataFrame([
        ['atp1d', 337, 411, 6, None],
        ['atp7d', 296, 411, 6, None],
        ['oes97', 334, 263, 16, None],
        ['oes10', 403, 298, 16, None],
        ['rf1', 9125, 64, 8, None],
        ['rf2', 9125, 576, 8, None],
        ['scm1d', 9803, 280, 16, None],
        ['scm20d', 8966, 61, 16, None],
        ['edm', 154, 16, 2, None],
        ['sf1', 323, 10, 3, None],
        ['sf2', 1066, 10, 3, None],
        ['jura', 359, 15, 3, None],
        ['wq', 1060, 16, 14, None],
        ['enb', 768, 8, 2, None],
        ['slump', 103, 7, 3, None],
        ['andro', 49, 30, 6, None],
        ['osales', 639, 413, 12, None],
        ['scfp', 1137, 23, 3, None],
    ], columns=['name', '#instance', '#instance_features', '#targets', '#target_features'])
    ,
    'multi-task learning': pd.DataFrame([
        ['dog', 800, '3*224*224', 52, None],
    ], columns=['name', '#instance', '#instance_features', '#targets', '#target_features'])
    ,
    'matrix completion': pd.DataFrame([
        ['ml-100k', 1000, None, 1700, None],
    ], columns=['name', '#instance', '#instance_features', '#targets', '#target_features'])
    ,
    'dyadic prediction': pd.DataFrame([
        ['srn', 1821, 9884, 113, 1685],
        ['ern', 1164, 445, 154, 445],
        ['dpie', 664, 664, 445, 445],
        ['dpii', 204, 204, 210, 210],
    ], columns=['name', '#instance', '#instance_features', '#targets', '#target_features'])
    }

    return info_per_mtp_setting