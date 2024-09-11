import pickle
import streamlit as st
import numpy as np

# Custom CSS for background color, font style, header, layout, and components
st.markdown(
    """
    <style>
    /* Change the background color of the main page */
    .main {
        background-image: url("/Users/leewen/Downloads/Recommendersystem/bg.png");
    }
    
    /* Customize header */
    h1 {
        color: #333333;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    
    /* Customize selectbox */
    .stSelectbox div[data-baseweb="select"] {
        font-size: 18px;
        font-weight: bold;
    }
    
    /* Customize the search bar */
    .stTextInput input {
        background-color: #f5f5f5;
        border: 2px solid #bcaaa4;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    
    /* Customize the button */
    .stButton button {
        background-color: #bcaaa4;
        color: white;
        border: none;
        padding: 8px 16px;
        font-size: 16px;
        border-radius: 40px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #a1887f;
    }
    
    /* Style for recommendation text */
    .recommendation-text {
        font-size: 16px;
        font-weight: bold;
        color: #333;
        text-align: center;
        font-family: 'Verdana', sans-serif;
    }

    /* Add a box around the image and text */
    .box {
        background-color: #ffffff;
        border: 2px solid #bcaaa4;
        padding: 10px;
        border-radius: 10px;
        margin: 10px; /* Center and space out the boxes */
        text-align: center;
        width: 160px; /* Fixed width */
        height: 300px; /* Fixed height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    /* Set fixed size for images */
    .box img {
        width: 150px;  /* Set image width */
        height: 220px; /* Set image height */
        object-fit: cover;
        margin-bottom: 5px; /* Add margin between image and title */
    }

    /* Center the text inside the box */
    .box strong {
        font-size: 14px;
    }

    /* Adjust the layout of columns for image display */
    .stColumns {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end; /* Align columns to the right */
        gap: 1rem; /* Space between columns */
        width: 100%; /* Ensure the container spans the full width */
    }

    /* Individual column container styling */
    .stColumn {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 350px; /* Set the height for better alignment */
    }
    .st-emotion-cache-1n76uvr {
    width: 704px;
    position: relative;
    display: flex;
    flex: 1 1 0%;
    flex-direction: column;
    gap: 1rem;
    margin-left: -40%;

    .st-emotion-cache-ocqkz7 {
    display: flex;
    flex-wrap: wrap;
    -webkit-box-flex: 1;
    flex-grow: 1;
    -webkit-box-align: stretch;
    align-items: stretch;
    gap: 1rem;
    margin-right: -65%;
    </style>
    """,
    unsafe_allow_html=True
)

st.header('Book Recommender System')

# Loading the pre-trained model and data
model = pickle.load(open('artifacts/model.pkl','rb'))
book_names = pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url


def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url


# Select a book from the dropdown
selected_books = st.selectbox(
    "Please type a book name",
    book_names
)

# Show recommendations when the button is clicked
if st.button('Show'):
    recommended_books, poster_url = recommend_book(selected_books)

    # Display recommendations in columns
    st.markdown("<div class='stColumns'>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"<div class='box'><strong>{recommended_books[1]}</strong><br><img src='{poster_url[1]}'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='box'><strong>{recommended_books[2]}</strong><br><img src='{poster_url[2]}'></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='box'><strong>{recommended_books[3]}</strong><br><img src='{poster_url[3]}'></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='box'><strong>{recommended_books[4]}</strong><br><img src='{poster_url[4]}'></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='box'><strong>{recommended_books[5]}</strong><br><img src='{poster_url[5]}'></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
