import streamlit as st
import pandas as pd
import pickle
import sys                                    
import requests 
from PIL import Image
import PIL.Image
import time 
import bs4 as bs
import urllib.request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

hide_st_style = """
            <style>
          
            footer {visibility: hidden;}
            background-color : red;

            div.
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
def recommend(movie): #recommend 5 movies based on a given movie
    
    a_1 = np.array(latent_matrix_1_df.loc[movie]).reshape(1,-1)
    a_2 = np.array(latent_matrix_2_df.loc[movie]).reshape(1,-1)
      #calculate similarity for this movie with other movies
    score_1 = cosine_similarity(latent_matrix_1_df,a_1).reshape(-1)
    score_2 = cosine_similarity(latent_matrix_2_df,a_2).reshape(-1)
      #avg of both content and collaborative 
    hybrid = ((score_1+score_2)/2.0)

    dictDf = {'hybrid':hybrid}
      #dict_df = {'hybrid':hybrid}
    similar = pd.DataFrame(dictDf,index=latent_matrix_1_df.index)
    similar.sort_values('hybrid',ascending = False, inplace = True)
    return similar[1:].head(8).index.tolist()






latent_matrix_1_df = pickle.load(open('l1.pickle','rb'))
latent_matrix_2_df = pickle.load(open('l2.pickle','rb'))
Final = pickle.load(open('movie_list.pkl','rb')) 
movies = pickle.load(open('genres.pkl','rb')) 
rate = pickle.load(open('rate_movie.pkl','rb'))





col1,col2,col3,col4,col5=st.columns(5)
with col1:
    img1 = Image.open('images/logo2.jpeg')
    img1 = img1.resize((300,250),)
    st.image(img1,use_column_width=False)
with col2:   
    img1 = Image.open('images/logo2.jpeg')
    img1 = img1.resize((300,250),)
    st.image(img1,use_column_width=False)
with col3:   
    img1 = Image.open('images/logo2.jpeg')
    img1 = img1.resize((300,250),)
    st.image(img1,use_column_width=False)
with col4:   
    img1 = Image.open('images/logo2.jpeg')
    img1 = img1.resize((270,250),)
    st.image(img1,use_column_width=False)

   

st.title('Movie Recommendation System')


movie_list = Final['title'].values
movie_genre = movies['genres'].values
movie_rate = rate['rating'].values

st.sidebar.title("Movie recommendation")
selected_movie = st.sidebar.selectbox(
"Type or select a movie from the dropdown",
movie_list)



if st.sidebar.button('Show Recommendation'):
    with st.spinner('Wait for it...'):
        time.sleep(2)
    
    st.caption('Recommendations:')
        
    names=recommend(selected_movie)
     
    col1,col2,col3,col4,col5,col6,col7,col8= st.columns(8)
    with col1:
        st.markdown(names[0])
    with col2:
        st.markdown(names[1])
    with col3:
        st.markdown(names[2])
    with col4:
        st.markdown(names[3])
    with col5:
        st.markdown(names[4])
    with col6:
        st.markdown(names[5])
    with col7:
        st.markdown(names[6])

            
    col1,col2,col3,col4,col5,col6,col7,col8= st.columns(8)
    with col1:
        
        st.write("rating:",movie_rate[0])
        with st.expander("genre"):
            st.write("genre:",movie_genre[0])
            
    with col2:
        
        st.write("rating:",movie_rate[1])
        with st.expander("genre"):
            st.write("genre:", movie_genre[1])
            
    with col3:
        
        st.write("rating:",movie_rate[2])
        with st.expander("genre"):
            st.write("genre:", movie_genre[2])
            
    with col4:
        
        st.write("rating:",movie_rate[3])
        with st.expander("genre"):
            st.write("genre:", movie_genre[3])
            
    with col5:
        
        st.write("rating:",movie_rate[4])
        with st.expander("genre"):
            st.write("genre:", movie_genre[4])
            
    with col6:
        
        st.write("rating:",movie_rate[5])
        with st.expander("genre"):
            st.write("genre:", movie_genre[5])
            
    with col7:

        st.write("rating:",movie_rate[6])
        with st.expander("genre"):
            st.write("genre:", movie_genre[6])
            

            
                
st.markdown("**------------------------------------------------------------------------TOP MOVIES------------------------------------------------------------------------------**")
         
    
col1, col2, col3,col4,col5,col6,col7,col8 =st.columns(8) #,col2,col3,col4,col5
with col1:
 
    img1 = Image.open('images/up.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)   
    #st.write("rating:",4.5)
    with st.expander("details"):
        st.write("rating:",5)
        st.write("language:" ,"English, Korean" )
        st.write("genre: " , "Adventure|Animation|Children|Drama") 
            
with col2:
    img1 = Image.open('images/transformers.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)   
    with st.expander("details"):
        st.write("rating:",4.5)
        st.write("language:" ,"English" )
        st.write("genre: " , "Action|Adventure|Sci-Fi|IMAX") 
            

with col3 : 
    img1 = Image.open('images/district.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)
    with st.expander("details"):
        st.write("rating:",5.0)
        st.write("language:" ,"English" )
        st.write("genre: " , "Mystery|Sci-Fi|Thriller") 
        
with col4 : 
    img1 = Image.open('images/hairspray.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)
    with st.expander("details"):
        st.write("rating:",4.0)
        st.write("language:" ,"English" )
        st.write("genre: " , "Comedy|Drama|Musical") 
        

with col1:
 
    img1 = Image.open('images/once.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)   
    #st.write("rating:",4.5)
    with st.expander("details"):
        st.write("rating:",5)
        st.write("language:" ,"English" )
        st.write("genre: " , "Drama|Musical|Romance")   
        
with col2:
 
    img1 = Image.open('images/ocean.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)   
    #st.write("rating:",4.5)
    with st.expander("details"):
        st.write("rating:",4.0)
        st.write("language:" ,"English" )
        st.write("genre: " , "Crime|Thriller")   
        
with col3 : 
    img1 = Image.open('images/fantastic.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)   
    #st.write("rating:",4.5)
    with st.expander("details"):
        st.write("rating:",4.0)
        st.write("language:" ,"English" )
        st.write("genre: " , "Action|Adventure|Sci-Fi") 
        
with col4 : 
    img1 = Image.open('images/shoes.png')
    img1 = img1.resize((145,230),)
    st.image(img1,use_column_width=False)   
    #st.write("rating:",4.5)
    with st.expander("details"):
        st.write("rating:",4.5)
        st.write("language:" ,"English" )
        st.write("genre: " , "Comedy|Drama") 
    
            
            
    
  

        
#imgN = Image.open('C:/Users/Hp/Downloads/movie_recommendation/movie-recommend-app/meta/logo.jpg')
#imgN = imgN.resize((250,250),)
        

