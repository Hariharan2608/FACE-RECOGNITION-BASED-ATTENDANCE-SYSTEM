import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import playsound
import base64
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# new_title = '<p style="font-family:times new roman; color:red; font-size: 42px;">IMAGE  CLASSIFICATION</p>'
# st.markdown(new_title, unsafe_allow_html=True)

main_bg = "bd1.jpg"
main_bg_ext = "jpg"


st.write(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}

    </style>
    """,
    unsafe_allow_html=True
)

home = st.sidebar.radio(
    label="Select",
    options=['Image Classification', 'Basic Image Processing','Advanced Image Processing','Face Detection'])
st.header((home))


if home == 'Image Classification':
    try:

        #st.write("Please upload the image by using browse files.")
        uploaded_file = st.file_uploader("Choose the Image ", type=["jpg","png","jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.image(image, caption='Uploaded Image.',width = 500)

            label = teachable_machine_classification(image, 'keras_model.h5')
        
            if label == 0:
                try:
                    st.success('The image that you has been uploaded  is "Andrews" ')
                    playsound.playsound("Audio/and.mpeg")
                except Exception as e:
                    print(e)
                
            if label == 1:
                try:
                    st.success('The image that you has been uploaded  is "Vishnuraja" ')
                    playsound.playsound("Audio/vishnu.mpeg")
                except Exception as e:
                    print(e)
                
            if label == 2:
                try:
                    st.success('The image that you has been uploaded  is "Praveen christopher" ')
                    playsound.playsound("Audio/chris.mpeg")
                except Exception as e:
                    print(e)
                    
            if label == 3:
                try:
                    st.success('The image that you has been uploaded  is "Hariharan" ')
                    playsound.playsound("Audio/hari.mpeg")
                except Exception as e:
                    print(e) 
            
            if label == 4:
                try:
                    st.success('The image that you has been uploaded  is "Kamalesh" ')
                    playsound.playsound("Audio/kamal.mpeg")

                except Exception as e:
                    print(e)
                    
            if label == 5:
                try:
                    st.success('The image that you has been uploaded  is "Sajan" ')
                    playsound.playsound("Audio/sajan.mpeg")
                except Exception as e:
                    print(e)
                    
        else:
            st.write("Please upload the image by using browse files.")
    except Exception as e:
        print(e)   
        
elif home == 'Basic Image Processing':
    try:
        st.write('In this "Basic Image Processing" you can do Grayscale,Resizing the image,Rotating the image,Cropping the image')
        uploaded_file = st.file_uploader("Choose an image")
        if uploaded_file is not None:
            image_upload = Image.open(uploaded_file)
            frame = np.array(image_upload)
            
            st.image(image_upload)
            
            choose = st.sidebar.selectbox(label = "select",options = ["Grayscale","Resize","Rotate","Crop"])
            st.subheader((choose))
            
            if choose == 'Grayscale':
                try:
                    st.write("In digital images, grayscale means that the value of each pixel represents only the intensity information of the light. Such images typically display only the darkest black to the brightest white. In other words, the image contains only black, white, and gray colors, in which gray has multiple levels.")
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    st.image(image)
                except Exception as e:
                    print(e)
                    
            elif choose == 'Resize':
                try:
                    st.write('When an image is resized, its pixel information is changed. For example, an image is reduced in size, any unneeded pixel information will be discarded by the photo editor')
                    st.write(frame.shape)
                    st.write("which is the dimension of your upload image. If you the height and width given in the side then you can resize the image.")
                    h1=st.sidebar.slider("Select the height", min_value=0,
                                                     max_value=1000, value=300)
                    w1=st.sidebar.slider("Select the width", min_value=0,
                                                     max_value=1000, value=400)
                    dimension = (w1, h1) 
                    resized_image = cv2.resize(frame, dimension, interpolation = cv2.INTER_AREA)
                    st.image(resized_image)
                except Exception as e:
                    print(e)
                    
            elif choose == 'Rotate':
                try:
                    st.write("When referring to an image or image editor, rotate is a feature that allows you to turn an image in a clockwise or counterclockwise direction.You can rotate the uploaded image by changing the degree which is given.")
                    (h1, w1) = frame.shape[:2]
                    center = (w1 / 2, h1 / 2)
                    degree = st.sidebar.slider("Select the Degree",min_value=0,
                                                     max_value=360, value=180)
                    Matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
                    rotated_image = cv2.warpAffine(frame, Matrix, (w1, h1))
                    st.image(rotated_image)
                except Exception as e:
                    print(e)
                    
            elif choose == 'Crop':
                try:
                    st.write(" Cropping is the removal of unwanted outer areas from a photographic or illustrated image. The process usually consists of the removal of some of the peripheral areas of an image to remove extraneous trash from the picture, to improve its framing, to change the aspect ratio, or to accentuate or isolate the subject matter from its background.")
                    st.write("###### You can crop the image by altering the ranges of height and width given in the sidebar.")
                    val = st.sidebar.slider("Select the height from top to bottom", min_value = 0,max_value = frame.shape[0],value = 0)
                    val_1 = st.sidebar.slider("Select the height from bottom to top", min_value = 0,max_value = frame.shape[1],value = frame.shape[1])
                    val2 = st.sidebar.slider("Select the width from left to right", min_value = 0,max_value = frame.shape[0],value = 0)
                    val2_1 = st.sidebar.slider("Select the width from right to left", min_value = 0,max_value = frame.shape[1],value = frame.shape[1])
                    cropped_image = frame[val:val_1,val2:val2_1]
                    st.image(cropped_image)
                except Exception as e:
                    print(e)
   
        else:
            st.write("Please upload the image by using browse files.")
    except Exception as e:
        print(e)

elif home == 'Advanced Image Processing':
    try:
        st.write('In this "Advanced Image Processing" you can alter the threshold of the image and find the edges in the image and also you can see the sobel edges')
        #st.write("Please upload the image by using browse files.")
        uploaded_file_1 = st.file_uploader("Choose the Image ", type=["jpg","png","jpeg"])

        if uploaded_file_1 is not None:
            image_1 = Image.open(uploaded_file_1)
            frame_1 = np.array(image_1)
        
            
            #st.image(image, caption='Uploaded Image.',width = 500)
            
            box = st.sidebar.selectbox(label = "Select",options = ["Threshold","Edge Detection","Sobel"])
            st.subheader((box))
            
            if box == "Threshold":
                try:
                    st.write("Threshold is to change the pixels of an image to make the image easier to analyze. In thresholding, we convert an image from color or grayscale into a binary image, i.e., one that is simply black and white. Most frequently, we use thresholding as a way to select areas of interest of an image, while ignoring the parts we are not concerned with. ")
                    st.image(image_1)
                    image = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
                    
                    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)
                    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
                    thresh1 = thresh1.astype(np.float64)
                    st.image(thresh1, use_column_width=True,clamp = True)
                    
                    st.write("#### Bar Chart of the image")
                    st.write("Histogram are constructed by splitting the range of the data into equal-sized bins (called classes). Then for each bin, the numbers of points from the data set that fall into each bin are counted. Vertical axis is Frequency (i.e., counts for each bin) and Horizontal axis is Response variable. In image histograms the pixels form the horizontal axis.")
                    histr = cv2.calcHist([image],[0],None,[256],[0,256])
                    st.bar_chart(histr)
                except Exception as e:
                    print(e)
                
            elif box == "Edge Detection":
                try:
                    st.write("Edge detection is a technique of image processing used to identify points in a digital image with discontinuities, simply to say, sharp changes in the image brightness. These points where the image brightness varies sharply are called the edges.In this applocation we use Canny edge detection. ")
                    st.image(frame_1)
                    canny_val = st.sidebar.slider("Threshold 1", min_value = 0,max_value = frame_1.shape[0],value = 50)
                    canny_val2 = st.sidebar.slider("Threshold 2", min_value = 0,max_value= frame_1.shape[1],value = 200)
                    edges = cv2.Canny(frame_1,canny_val,canny_val2)
                    st.image(edges,use_column_width=True,clamp=True)
                except Exception as e:
                    print(e)
                
            elif box == "Sobel":
                try:
                    st.write("The Sobel operator performs a 2-D spatial gradient measurement on an image and so emphasizes regions of high spatial frequency that correspond to edges. Typically it is used to find the approximate absolute gradient magnitude at each point in an input grayscale image.")
                    st.image(image_1)
                    # Convert to graycsale
                    img_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
                    # Blur the image for better edge detection
                    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
                    
                    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
                    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
                    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
                    
                    st.subheader("Sobel X")
                    st.image(sobelx,use_column_width=True,clamp=True)
                    
                    st.subheader("Sobel Y")
                    st.image(sobely,use_column_width=True,clamp=True)
                    
                    st.subheader("Sobel XY")
                    st.image(sobelxy,use_column_width=True,clamp=True)
                except Exception as e:
                    print(e)
                
        else:
            st.write("Please upload the image by using browse files.")
    except Exception as e:
        print(e)   
            
elif home == "Face Detection":
    try:          
        st.write("Face detection  also called facial detection  is an artificial intelligence (AI) based computer technology used to find and identify human faces in digital images. It now plays an important role as the first step in many key applications including face tracking, face analysis and facial recognition.")
        #st.write("Please upload the image by using browse files.")
        uploaded_file_2 = st.file_uploader("Choose the Image ", type=["jpg","png","jpeg"])
        if uploaded_file_2 is not None:
            image_2 = Image.open(uploaded_file_2)
            frame_2 = np.array(image_2)
            # Load the cascade
            face_cascade = cv2.CascadeClassifier("new 1.tek")
            # Read the input image
            #img = cv2.imread("C:/Users/Sajan/Music/project/sample/Andrews/19.jpg")
            st.write("#### Input Image")
            st.write("This is the input image the you have been uploaded.")
            st.image(image_2)
            
            # Convert into grayscale
            gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_2, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the output
            st.write(" #### Output Image")
            st.write("In this image you can see the rectangle which is detected for the face.")
            st.image(frame_2)
        else:
            st.write("Please upload the image by using browse files.")
            
    except Exception as e:
            print(e)