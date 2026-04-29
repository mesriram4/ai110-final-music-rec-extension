
1) Name of Original Project I expanded on: Music Recommender 
+ The Music Recommender aims to take a user profile and recommend new music that aligns 
with their preferences based on genre, mood, and energy. This progam adds specific weights to which 
factors should be prioritzed the most when determining top songs for specific users (eg. more weight 
on genre as a siginificant contributer to preferences). 

2) Title of the Project: Retrieval-Augmented Generation of Top Songs Based on User Prompts
+ Summary: While similar to purpose of the original Music Recommender, the implementation of AI 
into this project aims to diversify preferences for users. Instead prioritizing one specific 
characteristic to offer recommendations, users can customize recommendations (Eg. asking for 
the top 5 songs based on energy only or both energy and mood). By incorporating 
this update, users have more control over what songs they want to be recommended instead of 
letting the program prioritize songs for them. Overall, the integration of AI offers users more 
autonomy. Most importantly, output of the RAG doesn't always have to align with preferences listed 
by the user profile, allowing users to diversify their listening preferences even more.

3) Architecture Overview 
+ When working with Claude to determine how Music_RAG.py (housing AI application) should 
be structured, one particular structure that stood out to be is splitting the RAG into 
two phases: the first phase takes a list of songs, embeds them into vectors, then saves them t
to ChromeDB to later be accessed once the program takes in a user prompt. Phase 2 embeds 
the prompt itself into vectors, which are then compared to songs within the database, ultimately 
providing recommendations that align with the prompt. 

Here is the structure of the RAG: 

== Phase 1 == 
+ list_to_text_conv(csv_path): Reformat songs in csv file as labeled text, later appended
into the list 'descriptions'
+ embed_songs(descriptions): Take list of descriptions, embed them into vectors, then return 
those vectors 
+ store_vector(vectors, descriptions): Both vectors and corresponding descriptions are stored in 
ChromeDB

== Phase 2 ==
+ embed_prompt(prompt): We embed user's prompt into vectors 
+ nearest_songs(query_vector, collection, k: int = 5): Compares vector of prompt to vectors within 
ChromeDB and returns 5 songs that best match prompt vectors.
+ format_retrieved(results): Reformat retrieved songs to labelled text. 
+ generate_recs(prompt, formatted_songs): Runs the LLM (connects to GPT 4o-mini), communicates 
with the model through a message, and retrieves the model's response.