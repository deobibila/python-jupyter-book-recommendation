import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the dataset
@st.cache_data
def load_data():
    try:
        books = pd.read_csv('books.csv', on_bad_lines='skip')
        books.drop_duplicates(subset=['bookID'], inplace=True)
        books.fillna('', inplace=True)
        
        # Prepare metadata for recommendation
        books['metadata'] = (
            books['authors'] + ' ' +
            books['language_code'] + ' ' +
            books['text_reviews_count'].astype(str)
        )
        
        # Vectorize metadata
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(books['metadata'])
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Map titles to indices
        title_to_index = pd.Series(books.index, index=books['title']).drop_duplicates()
        
        return books, cosine_sim, title_to_index
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Recommendation function
def recommend_books(title, books, cosine_sim, title_to_index):
    if title in title_to_index:
        idx = title_to_index[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4]  # Top 3 recommendations
        book_indices = [i[0] for i in sim_scores]
        return books[['title', 'authors', 'average_rating']].iloc[book_indices]
    else:
        return pd.DataFrame()  # Return empty DataFrame if book not found

def main():
    st.title("📚 Book Recommendation System")
    st.write("Discover books you'll love!")

    # Load data
    books, cosine_sim, title_to_index = load_data()
    
    if books is None:
        st.error("Could not load the book dataset. Please check the 'books.csv' file.")
        return

    # User input for book recommendation
    st.header("Get Book Recommendations")
    book_input = st.text_input("Enter a book you enjoyed:")
    
    if book_input:
        recommendations = recommend_books(book_input, books, cosine_sim, title_to_index)
        
        if not recommendations.empty:
            st.subheader("Books You Might Enjoy:")
            st.dataframe(recommendations)
        else:
            st.warning("Book not found in our library. Try another title!")

    # Visualization Section
    st.header("Book Data Insights")
    
    # Rating Distribution Bar Chart
    st.subheader("Book Ratings Distribution")
    rating_counts = books['average_rating'].value_counts().head(10)
    st.bar_chart(rating_counts)

    # Ratings Count vs Average Rating Scatter Plot
    st.subheader("Ratings Count vs Average Rating")
    fig, ax = plt.subplots()
    ax.scatter(books['ratings_count'], books['average_rating'], alpha=0.5)
    ax.set_xlabel('Ratings Count')
    ax.set_ylabel('Average Rating')
    ax.set_title('Relationship between Ratings Count and Average Rating')
    st.pyplot(fig)

    # Top Authors Pie Chart
    st.subheader("Top Authors by Book Count")
    top_authors = books['authors'].value_counts().head(10)
    fig, ax = plt.subplots()
    ax.pie(top_authors, labels=top_authors.index, autopct='%1.1f%%')
    ax.set_title("Proportion of Books by Top Authors")
    st.pyplot(fig)

    # Log app access
    logging.info("Recommendation system accessed.")

if __name__ == "__main__":
    main()
