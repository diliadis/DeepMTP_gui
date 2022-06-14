import streamlit as st

personal_website_link = '''<a href='https://kermit.ugent.be/phd.php?author=D.%20Iliadis&year=' target='_blank'> <img src='https://raw.githubusercontent.com/diliadis/streamlit_deepMTP/master/images/ugent_favicon.png'> </a>'''
linkedin_link = '''<a href='https://gr.linkedin.com/in/dimitris-iliadis-a2bb4a113' target='_blank'> <img src='https://cdn.exclaimer.com/Handbook%20Images/linkedin-icon_32x32.png'> </a>'''
github_link = f'''<a href='https://github.com/diliadis' target='_blank'> <img src='github_icon.png'> </a>'''
google_colab_link = '[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fcrS_vAjjE56KUTYC4gseAW0rwKB7nMu?usp=sharing)'

st.title('About')
st.write('This is a supporting app for the deepMTP framework.')

st.write('## Related material')
st.write(f'deepMTP tutorial: {google_colab_link}', unsafe_allow_html=True)
st.write('###')
# st.write('Presentation on Multi-Target prediction + coding tutorial:')

video_link = r'''[Alternative link for tutorial video]
(http://kibit.eti.pg.gda.pl/AI/ISSonDL2021/Dimitrios%20Iliadis%20-%20Multi-target%20%20predictions.mp4)
'''

st.video('http://kibit.eti.pg.gda.pl/AI/ISSonDL2021/Dimitrios%20Iliadis%20-%20Multi-target%20%20predictions.mp4')
# st_player('http://kibit.eti.pg.gda.pl/AI/ISSonDL2021/Dimitrios%20Iliadis%20-%20Multi-target%20%20predictions.mp4')

st.write(video_link)
st.write('## Contact')
st.write(f'{personal_website_link} {linkedin_link}', unsafe_allow_html=True)

st.write('## Funding')
st.write('This research received funding from the Flemish Government under the “Onderzoeksprogramma Artificiele Intelligentie (AI) Vlaanderen” programme.')

st.write('***')
# my_file = 'https://drive.google.com/file/d/1kh4IixoOvSVru1awxo91eVSknIrq9iQd/view'
# video_file = open(my_file, 'rb').read()
# st.video(video_file)
# st.write('***')

col1, col2, col2a, col2b = st.columns(4)
with col1:
    # st.image('images/logo_UGent_EN_RGB_2400_color.png', width=200)
    st.image('images/ugent_screenshot_no_background.png', width=100)
with col2:
    # st.image('images/icon_UGent_BW_EN_RGB_2400_color.png', width=200)
    st.image('images/bio_screenshot_no_background.png', width=200)
with col2a:
    # st.image('images/icon_UGent_BW_EN_RGB_2400_color.png', width=200)
    st.write('')
with col2b:
    # st.image('images/icon_UGent_BW_EN_RGB_2400_color.png', width=200)
    st.write('')

col3, col4, col4a, col4b = st.columns(4)
with col3:
    # st.image('https://airesearchflanders.be/images/aiflanders-logo.png', width=200)
    st.image('images/ai_flanders_screenshot_v2.png', width=80)
with col4:
    # st.image('https://kermit.ugent.be/images/logo/kermit_logo.png', width=200)
    st.image('images/kermit_screenshot_no_background.png', width=110)
with col4a:
    # st.image('images/icon_UGent_BW_EN_RGB_2400_color.png', width=200)
    st.write('')
with col4b:
    # st.image('images/icon_UGent_BW_EN_RGB_2400_color.png', width=200)
    st.write('')