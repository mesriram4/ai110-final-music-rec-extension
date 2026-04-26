# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Model Name: Song Wizard Version 1

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

+ This music recommender is designed for anyone who wants to diversify their playlist while also sticking to elements reflective of their comfort zone and preferences in music. 

Prompts:  

- What kind of recommendations does it generate 

    This model will generate recommendations on your preferred genre, mood, and energy. 

- What assumptions does it make about the user  
    It will mostly make assumptions about user energy level. Whether they like energetic music or something a bit more mellow. This allows for genre crossing and expanding music taste. 

- Is this for real users or classroom exploration  

    This program can be used by both audiences, both for user convenience and for students to understand how recommendations are simulated as well as pitfalls associated with it. 

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.) : 

    Genre, energy, and mood were all used for each song. 

- What user preferences are considered  
    Energy is the most highly consdiered user preferece, with genre being the last. 

- How does the model turn those into a score  
    The model assigns a 1.0 or 0.0 to songs that do or don't match either one of the user preferences. If a user preference matches, the song is awarded weighted points based on what feature should be prioritized. The equation, total = 0.20 * genre_score + 0.30 * mood_score + 0.40 * energy_score, is used to calculate the total score of a song, with the top preferred number being displayed. 
- What changes did you make from the starter logic  

    I decided to remove features I didn't need for building this recommender, such as acousticness. 

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
    + There were 20 songs in the catalog

- What genres or moods are represented  
    + Genres represeted as pop, lofi, jazz and r and b. Moods corresponded to some of these genres and ranged from sad to optimistic. 

- Did you add or remove data  
    I added additional data to strengthen the dataset and song options the software can recommend. 

- Are there parts of musical taste missing in the dataset  
    If there are more niche genres or interesting moods to use to expand the vocabulary or breath of knowledge of music in the dataset, then this recommender can better pick on music that the user will be more interested in. 

---

## 5. Strengths  

Where does your system seem to work well  

Consistent, lack of bugs, very capable of recommended music that closely aligns with energy and mood. 

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider: 

    There are multiple features excluded from the scoring system that are assumed to correspond to primary features or are considered irrelevant to the song selection prcoess. 

- Genres or moods that are underrepresented 

    This especially happens when niche genres are recommended. For example, when I included, "Shoegaze", this genre was so niche that there wasn't matching data. However, there were moods like "dreamy" that were used as the next best method for recommending songs for this genre, since shoegaze also fits this mood. Interstingly enough, this "dreamy" song is actually a country song, which doesn't necessarily align with the taste of many shoegaze fans. 

- Cases where the system overfits to one preference: 

    + This happened especially with adversarial profiles. If genre doesn't match, the second most strongly weighted feature will be take into account. Initially, mood was the second most strongly weighted feature, therefore, songs that don't match in genre or mood will only be compared by energy. For example, shoegaze had only one other matching mood in the datafile: "dreamy". This song will consistently rank the highest, while the rest of the songs in the list have to purely be evaluated for having similar energy to shoegaze music. 

- Ways the scoring might unintentionally favor some users: 

    Users who are into mainstream or more recognizable genres are more likely to benefit from this scoring system compared to users who listen to more alternative or niche genres. 

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested : 

        I tested three user profiles. One with a mainstream genre (lofi) and two with more niche genres (adversarial user profiles; shoegaze and bebop jazz). 
- What you looked for in the recommendations: 

    I looked not only for accuracy in genre and mood, but I also looked for differences in files that generate one type of ranking vs. the other based on differences in energy, which impacts score total. 

- What surprised you: 
    + What surprised me is that despite differences in scores for the same song for one run, the score will change, but the ranking will still remain the same. 

- Any simple tests or comparisons you ran  
    + As recommended, I switched the weight of genre and energy to test the impact of these features on the progrma, but I still ended up with similar results. For niche genres, there is a lack of change in rankings compared to main stream genres like lofi. 

No need for numeric metrics unless you created some.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
