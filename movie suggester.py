#!/usr/bin/env python3
"""
Movie Suggesting made easy by python

This script creates a basic Movie Recommender System using collaborative filtering.
It uses a subset of the MovieLens dataset and suggests movies based on user ratings.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend, which is widely compatible
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')  # Ignoring warnings for cleaner output
sns.set_style('white')  # Set seaborn style for plots

# Load the user ratings data into a Pandas DataFrame
# dataset.csv should contain columns: user_id, item_id, rating, timestamp separated by tab
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('dataset.csv', sep='\t', names=column_names)
print("User ratings data sample:")
print(df.head())  # Show sample of user ratings data

# Load the movie titles and their IDs
# movieIdTitles.csv should contain item_id and title columns
movie_titles = pd.read_csv('movieIdTitles.csv')
print("\nMovie titles data sample:")
print(movie_titles.head())  # Show sample of movie titles data

# Merge ratings with movie titles on 'item_id' for easier analysis
df = pd.merge(df, movie_titles, on='item_id')
print("\nMerged data sample:")
print(df.head())  # Show sample of merged data

# Calculate average rating for each movie
movie_mean_ratings = df.groupby('title')['rating'].mean().sort_values(ascending=False)
print("\nTop 5 movies by average rating:")
print(movie_mean_ratings.head())

# Calculate number of ratings each movie received
movie_rating_counts = df.groupby('title')['rating'].count().sort_values(ascending=False)
print("\nTop 5 movies by number of ratings:")
print(movie_rating_counts.head())

# Create a DataFrame with average rating and number of ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['numOfRatings'] = df.groupby('title')['rating'].count()
print("\nRatings DataFrame sample:")
print(ratings.head())

# Visualize distribution of number of ratings per movie
plt.figure(figsize=(10, 4))
ratings['numOfRatings'].hist(bins=70)
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Movies')
plt.title('Distribution of Number of Ratings')
plt.show()

# Visualize distribution of average ratings
plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)
plt.xlabel('Average Rating')
plt.ylabel('Count of Movies')
plt.title('Distribution of Average Ratings')
plt.show()

# Visualize joint relationship between average rating and number of ratings
sns.jointplot(x='rating', y='numOfRatings', data=ratings, alpha=0.5)
plt.show()

# Create the user-movie rating matrix for collaborative filtering
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

print("\nUser-movie rating matrix sample:")
print(moviemat.head())


# Function to get movie recommendations based on a movie title input
def get_movie_recommendations(movie_name, moviemat=moviemat, ratings=ratings):
    if movie_name not in moviemat:
        print(f"Movie '{movie_name}' not found in database.")
        return []

    # User ratings for the input movie
    movie_ratings = moviemat[movie_name]

    # Compute correlation of this movie's ratings with all others
    similar_to_movie = moviemat.corrwith(movie_ratings)

    # Create DataFrame for correlations
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)

    # Join with number of ratings to filter on popularity
    corr_movie = corr_movie.join(ratings['numOfRatings'])

    # Filter movies with at least 50 ratings to avoid noise
    recommendations = corr_movie[corr_movie['numOfRatings'] > 50].sort_values(by='Correlation', ascending=False)

    # Remove the movie itself from recommendations
    recommendations = recommendations[recommendations.index != movie_name]

    return recommendations


# Example usage: input movie and get top 5 recommended movies
input_movie = "Four Rooms (1995)"  # Change this value to test other movies
recommendations = get_movie_recommendations(input_movie)

print(f"\nTop recommendations based on '{input_movie}':")
print(recommendations.head(5))

# Optional: Save recommendations to CSV file
recommendations.head(10).to_csv('top_recommendations.csv')

