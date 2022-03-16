"""
k - Nearest Neighbours  (k-NN)

    Use the movie_lens data to guess the rating of a movie,
        by looking at the 10 movies closest to it in genre
        and popularity.
    How about the closest 15 movies?

    STEPS:
        1. Import columns and create a dataframe
        2. Define a function to calculate the distance
            between 2 movies
        3. Define a function to calculate the k-NN of your movie

"""

import numpy as np
import pandas as pd
from scipy import spatial
import operator


"""
1. Load dataset
"""
path = "file path/u.data"

"""
1.1 Add column headers
 
    After examining the data details, we see that there are
        no column names for columns, so pandas uses the first
        row in each column as the name for the column.
        Let us deal with this.
    Note that we just need the first three columns, and not 
        the last one, so we would specify this with the 
        ' usecols= ' attribute.
"""

column_names = ['user_id', 'movie_id', 'rating']

"""
1.2 Specify the type of separation, if it is not through 
    comas.
    
    Since the data is tab-separated, we need to specify this
        while reading the '.csv' file, using the code snippet:
        "   sep='\t'  "
"""
ratings = pd.read_csv(path,
                      sep='\t',
                      names=column_names,
                      usecols=range(3))

"""
2. See data details
"""


def data_details(name_of_dataframe):
    print('     First Five Rows: ')
    print('')
    print(name_of_dataframe.head())
    print('')

    print('     Last Five Rows: ')
    print('')
    print(name_of_dataframe.tail())      # to see how many rows we have
    print('')

    print('     Total Information about the dataset: ')
    print('')
    print(name_of_dataframe.info())    # understand our data
    print('')

    print('     Description of the dataset: ')
    print('')
    print(name_of_dataframe.describe())
    print('')


# data_details(ratings)

"""
3. Grouping
    We would group all the information by movie_id, then try to 
        calculate the total number of ratings.
    Once we have that, we would calculate the average rating for 
    each movie.
    
    For that, we would apply the 'group_by' method on the 'ratings'\
        dataframe, then we would aggregate the results by 
        'rating size' and 'mean rating'.
        
        We would then look at the data details for this new result.
"""
movie_properties = ratings.groupby('movie_id').agg(
    {'rating': [np.size, np.mean]})  # rating size' and 'mean rating'

# data_details(movie_properties)


# Example of how to read the movie_properties data:
#       "421 people (rating size) rated movie 1 (movie_id), and
#       the mean rating for movie 1 was 3.87".


"""
4. Let us normalize the number of ratings (rating size) so
    that it can be between 0 and 1.
        0 means no one rated the movie.
        1 means the movie had the highest rating compared to all 
            other movies.
            
        This normalization would help balance the data.
"""

movie_num_ratings = pd.DataFrame(movie_properties['rating']['size'])

movie_normalized_num_ratings = movie_num_ratings.apply(
    lambda x: (x - np.mean(x))/(np.max(x) - np.min(x)))

# data_details(movie_normalized_num_ratings)


"""
5. We would now look at another file: 'u.item'

    - This file contains info on the genre of movies.
    - Here, each movie is rated as '0' or '1' against a 
        certain genre, 0 being that it is not in that genre, 
        and 1 being that it is in that genre.
"""

movie_dict = {}     # construct an empty movie dictionary

# please note the 'r' before the path description
path_2 = r"C:/Users/HP/PycharmProjects/MachineLearningEnow/" \
       "Practical ML in Python/Notes-master/u.item"

while True:
    f = open(path_2)
    temp = ''

    for line in f:
        fields = line.rstrip('\n').split('|')
        movie_id = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        # genres = map(int, genres)
        # The line of code above blocked the info from
        # showing,reason why it was commented out.

        # Construct a dictionary which
        # maps the movie_id to the name(1), genre(2)
        # and size(3) and ratings data(4)
        movie_dict[movie_id] = (name,
                                genres,
                                movie_normalized_num_ratings.loc[
                                    movie_id].get('size'),
                                movie_properties.loc[
                                    movie_id].rating)
    break


# Lets see what information we have for certain movies
# we define a function to access movie information
# from 'movie_dict' given the an index number

def movie_info(position):
    for x in range(0, 4):
        print(movie_dict[position][x])


# movie_info(1)


"""
6. We now define a function to calculate the distance 
    between two movies on the bases of:
    1. Popularity ratings
    2. Genres
"""


def compute_distance(a, b):

    # we would calculate the distance for the genres
    # using the cosine similarity function
    """
    genres_a = a[1]
    genres_b = b[1]
    genre_distance = spatial.distance.cosine(genres_a, genres_b)
    """

    # we then calculate the absolute distance for
    # popularity ratings and store in a variable called
    # 'popularity_distance'.
    popularity_a = a[2]
    popularity_b = b[2]
    popularity_distance = abs(popularity_a - popularity_b)

    # compute the total distance
    """
    total_distance = genre_distance + popularity_distance
    """

    return popularity_distance


"""Only the popularity distance was used"""
# compute_distance(movie_dict[1], movie_dict[2])


"""
7. We now define a function to calculate the k-NN of our movie
"""


def get_neighbours(movie_id, k):

    """
    - Note that 'movie_ide' (the positional argument above)
        is our 'movie_id' variable.
    - We altered the spelling in order not to conflict
        the name with our 'movie_id' in the code.
    """

    distances = []

    for movie in movie_dict:
        if movie != movie_id:
            dist = compute_distance(
                movie_dict[movie_id], movie_dict[movie])

            distances.append((movie, dist))

    distances.sort(key=operator.itemgetter(1))

    neighbours = []

    for i in range(k):
        neighbours.append(distances[i][0])

    return neighbours


"""
7b. Get neighbour information
"""


def get_neighbour_information(k, average_rating):

    neighbours = get_neighbours(1, k)

    for neighbour in neighbours:
        average_rating += movie_dict[neighbour][3]
        print(
            movie_dict[neighbour][0] + " " + str(
                movie_dict[neighbour][3]))

    # What are these next two lines of code for?

    average_rating /= float(k)

    print(average_rating)


"""
you can change the k value and average ratings in order 
to see other results
"""

# get_neighbour_information(10, 0)
