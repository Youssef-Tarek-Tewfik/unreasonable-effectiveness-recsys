INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file anime.inter comprising the ratings of users over the anime.
Each record/line in the file has the following fields: user_id, item_id, rating

user_id: the id of the users and its type is token. 
item_id: the id of the anime and its type is token.
rating: rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating), and its type is float.

ANIME INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file anime.item comprising the attributes of the anime.
Each record/line in the file has the following fields: item_id, name, genre, type, episodes, avg_rating, members
 
item_id: the id of the anime and its type is token.
name: full name of anime, and its type is token_seq.
genre: comma separated list of genres for this anime, and its type is token_seq.
type: the type of the animes, such as movie, TV, OVA, etc, and its type is token.
episodes: how many episodes in this show (1 if movie), and its type is float.
avg_rating: average rating out of 10 for this anime, and its type is float.
members: number of community members that are in this anime's "group", and its type is float.
