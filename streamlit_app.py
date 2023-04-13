#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%writefile breast-cancer-imaging.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

html_temp = """ 
  <div style="background-color:pink ;padding:8px">
  <h2 style="color:white;text-align:center;"><b>INTEGRATED DEEP LEARNING FRAMEWORK FOR BREAST CANCER IDENTIFICATION USING MEDICAL IMAGES<b></h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)
activities=['Digital Infrared Thermal Imaging (DITI)','Ultrasound','Digital Mammography (DM)','Digital Breast Tomosynthesis (DBT)','Dynamic Contrast Enhanced Magnetic Resonance Imaging (DCE-MRI)','Biopsy= basic confirmation','Biopsy= IDC confirmation']
dia_opt=st.sidebar.selectbox('Select any one of the above',activities)
if dia_opt=='Digital Infrared Thermal Imaging (DITI)':
            st.subheader('This tool is to diagnose and predict for the presence of breast cancer based on thermal imaging.')
            st.subheader('The user is requested to upload the infrared image of the breast.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"breast thermal classification (2.4).h5")
                return model
            with st.spinner('Model is being loaded..'):
                thermal_model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=False)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = thermal_model.predict(imaga)
                a=predictions[0]
                if st.button('Click to get results:'):
                    if a<0.50:
                        st.success('The subject under consideration is observed to be NORMAL.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.error('The subject under consideration is suspected to have breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
elif dia_opt=='Ultrasound':
            st.subheader('This tool is to diagnose and predict for the presence of breast cancer based on ultrasound imaging.')
            st.subheader('The user is requested to upload the ultrasound image of the breast.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"breast ultrasound classification (2.4).h5")
                return model
            with st.spinner('Model is being loaded..'):
                ultra_model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=False)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = ultra_model.predict(imaga)
                a=predictions[0]
                if st.button('Click to get results:'):
                    #st.write('Confidence score'+a)
                    if a<0.50:
                        st.success('The subject under consideration is suspected to have BENIGN breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.error('The subject under consideration is suspected to have MALIGNANT breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
elif dia_opt=='Digital Mammography (DM)':
            st.subheader('This tool is to diagnose and predict for the presence of breast cancer based on mammogram imaging.')
            st.subheader('The user is requested to upload the mammogram image of the breast.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"mammo BM prediction pgm data (2.4).h5")
                return model
            with st.spinner('Model is being loaded..'):
                mammo_model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=True)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = mammo_model.predict(imaga)
                a=predictions[0][1]
                if st.button('Click to get results:'):
                    if a<0.50:
                        st.success('The subject under consideration is suspected to have BENIGN breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.error('The subject under consideration is suspected to have MALIGNANT breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
elif dia_opt=='Digital Breast Tomosynthesis (DBT)':
            st.subheader('This tool is to diagnose for breast abnormalities based on DBT images.')
            st.subheader('The user is requested to upload the bmp version of the DBT image.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"DBT normal abnormal (2.4).h5")
                return model
            with st.spinner('Model is being loaded..'):
                model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=False)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = model.predict(imaga)
                a=predictions[0]
                if st.button('Click to get the results:'):
                    if a>0.50:
                        st.success('The subject under observation appears to be normal.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                #st.error('The subject under consideration is suspected to have breast abnormalities. There are chances that the abnormality is breast cancer. Please ensure that you consult with an oncologist for further diagnosis.')
                else:
                        st.error('The subject under consideration is suspected to have breast abnormalities. There are chances that the abnormality is breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
            if st.button('Click to get the severity'):
                    @st.cache(allow_output_mutation=True)
                    def load_model():
                        model=tf.keras.models.load_model(r"mammo BM prediction pgm data (2.4).h5")
                        return model
                    with st.spinner('Model is being loaded..'):
                        model=load_model()
                        #file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
                        #st.set_option('deprecation.showfileUploaderEncoding', False)
                    img = Image.open(file)
                        #st.image(img, use_column_width=False)
                    size = (224,224)    
                    image = ImageOps.fit(img, size, Image.ANTIALIAS)
                    imag = np.asarray(image)
                    imaga = np.expand_dims(imag,axis=0) 
                    predictions = model.predict(imaga)
                    a=predictions[0]
                    if a.any()>0.50:
                        st.error('The subject under consideration is observed to be MALIGNANT.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.success('The subject under consideration is observed to be BENIGN.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
elif dia_opt=='Dynamic Contrast Enhanced Magnetic Resonance Imaging (DCE-MRI)':
            st.subheader('This tool is to diagnose and predict for the presence of breast cancer based on DCE-MRI imaging.')
            st.subheader('The user is requested to upload the DCE-MRI image of the breast.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"DCE breast cancer detection.h5")
                return model
            with st.spinner('Model is being loaded..'):
                dce_model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=False)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = dce_model.predict(imaga)
                a=predictions[0]
                if st.button('Click to get results:'):
                    #st.write('Confidence score'+a)
                    if a<0.50:
                        st.success('The subject under consideration has BENIGN breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.error('The subject under consideration has MALIGNANT breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
elif dia_opt=='Biopsy= basic confirmation':
            st.subheader('This tool is to diagnose and predict for the presence of breast cancer based on histopathological imaging.')
            st.subheader('The user is requested to upload the histopathological image of the breast obtained from biopsy.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"biopsy-basic.h5")
                return model
            with st.spinner('Model is being loaded..'):
                bio_model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=False)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = bio_model.predict(imaga)
                a=predictions[0]
                if st.button('Click to get results:'):
                    #st.write('Confidence score'+a)
                    if a<0.50:
                        st.success('The subject under consideration has BENIGN breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.error('The subject under consideration has MALIGNANT breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
elif dia_opt=='Biopsy= IDC confirmation':
            st.subheader('This tool is to diagnose and predict for the presence of invasive ductal carcinoma (IDC) based on biopsy imaging.')
            st.subheader('The user is requested to upload the histopathological image of the breast obtained from biopsy.')
            @st.cache(allow_output_mutation=True)
            def load_model():
                model=tf.keras.models.load_model(r"IDC-biopsy.h5")
                return model
            with st.spinner('Model is being loaded..'):
                idc_model=load_model()
            file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is None:
                 st.text("Please upload an image file within the allotted file size")
            else:
                img = Image.open(file)
                st.image(img, use_column_width=False)
                size = (224,224)    
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                imag = np.asarray(image)
                imaga = np.expand_dims(imag,axis=0) 
                predictions = idc_model.predict(imaga)
                a=predictions[0]
                if st.button('Click to get results:'):
                    #st.write('Confidence score'+a)
                    if a<0.50:
                        st.success('The subject under consideration deos not have IDC breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
                    else:
                        st.error('The subject under consideration has IDC breast cancer.')
                        b=round(np.amax(predictions) * 100, 2)
                        st.write("confidence score "+str(b))
st.subheader('TEAM MEMBERS')
st.success('1. V.A.SAIRAM')
st.success('2. SAMYUKTHA KAPOOR')
st.success('3. J.NITHILA')

