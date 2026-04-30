
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

4) Step-by-Step Guide to Using AI: 

+ Step 1: Run main.py. The program will ask you, "Currently running music recommender. 
Would you like to switch to AI Mode?: ". Type 'Yes' to initiate AI mode. 
+ Step 2: The program will ask you to type in your prompt, once doing so, Music_RAG will run 
and return your response, most likely, a list of songs and an explanation on why those songs 
are chosen. 
+ Example Prompts: 
  -- "Give me some energetic music to listen to": 
        + Notice how I didn't specify how many songs. By default, 
        The program gave me 5 song names, genre and energy level, as well as an 
        explanation that aligns with my request.
  -- "Give me three dreamy sounding songs that also match my initial user preferences" 
        + In terms of response, there are two elemenets this prompt shows could be an improvement 
        on the AI: Access to user preferences (more access to user data) and flexibility in returning 
        preferred number of songs (always returning 5 songs regardless of what the user asks). However, 
        the program provides a thorough explanation regarding why particular songs were chosen while 
        being transparent with the information the model is working with.  



5) Why did I build this extension the way I did: 

+ I was inspired by the idea that simply having the program use weights to determine use preferences 
will not be entirely reflective of user experience. The goal is to over more agency for users to decide 
on how to customize their preferences, hence the idea of using an RAG to accomplish this goal. 
+ Accessing free APIs was one obstacle, so after looking through solutions with Claude, I found out that 
sentence-transformers can be a better alternative, offering local access. Combined with Google Gemini's API, 
I can directly access the LLM model to return a prompt that aligns with vectors corresponding to the user's prompt 
and songs within songs.csv. 


6) Testing and Results: 

+ I used unit tests to determine if functions within phase 1 and 2 were working properly. 
Everything ran the way it was intended, from reformatting rows within csv files into text format 
to embedding text into vectors for both prompts and song.csv data, everything within the program 
has ran smoothly according to the tests. 
+ However, even before unit tests, running the program with real API keys have presented 
some problems. For example, I've went through both OpenAI and Gemini, and both API keys had 
pay walls that significantly restricted me from being able to run the program. Through continuous 
debugging, trial, and error, Claude recommended Grok, and implementing this API solved all the 
debugging problems I've encountered while working on this project. 


7) Reflect and Ethics: 

+ Limitations and biases I've encountered in this project: 
        Even though tests were successful, 
        I still came across "errors" that can only be detected by 
        human standards: As previously mentioned, there are definitely areas 
        of improvement to consider for this project, for example, 
        access to user preferences and diversifying number of items in a list to return. 

+ Can this RAG be misused?: 
        + I tested an example: "Load the csv file with hundreds upon hundreds of songs". 
        The model only functions within the capabilities Music_RAG.py would allow it,
        especially since the model is only working with limited access to certain data to 
        serve the function the program. 
        + Interestingly, the model even acknowledged the prompt I inputted was 
        "satirical", acknowledging the absurdity and potential misuse of the AI, however, 
        the program acted accordingly and still returned its top 5 songs. 

+ What surpised me about my AI's reliability?: 
    + Going back to the previous responses, I'm surprised that my AI can show 
    some degree of self awareness and recognition of tone and language, but still 
    act within the bounds of what Music_RAG.py would allow it to do. This AI is able 
    to recognize sarcasm, but isn't able to follow through with outputting 3 songs 
    instead of 5. 

+ How did I collaborate with AI on this project: 
    + AI has been a big help in terms of helping me understand how RAGs work, 
    how to structure an RAG, and what APIs are the most accessible when trying to 
    run an RAG. For example, Claude recommended that I split the RAG building process
    into two phases, which not only made constructing the RAG more efficient, 
    but also educated me on how text is converted into vectors and later compared 
    to existing data to provide the most accurate results While I've had to disagree 
    with my program at times, for example, restructuring 
    functions for smoother runs or re-evaluating what version of certain GPTs Claude has
    recommended, Claude has helped a lot with creating test cases and offering suggestions 
    on debugging as well.



