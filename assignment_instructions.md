In this assignment, you will explore the Goodreads Dataset to find insights, build models for making book recommendations, add an AI-powered personalization layer, and submit a slide deck of your findings along with a short video demo of your application. While you are free to use any software (such as Excel, Tableau, etc.) for data exploration and/or visualization, the modeling and app must be coded in Python. You are expected to submit your code, so try to write clean and well-commented code for the instructor's review and for your future reference.

For this project, you can work individually or in self-selected teams of two (or three). You should sign up for your group under Projects 2&3 Groups no later than Monday 06/08. Please note that team members need to be from the same section and only one submission per team through Canvas is sufficient. NOTE: If you submit your work individually for project 2, you should do the same for project 3, and if you submit your project 2 as a group, you should do the same for project 3. You cannot switch your team for projects 2 and 3. 

 

Key Requirements
Explore the dataset and provide any insights you find about users, reviews, and the books. This is an important step as unsupervised learning projects are exploratory by nature. Highlight patterns, anomalies, or questions the data raises that could affect modeling or recommendations.
Compare and evaluate a set of recommendation models based on user-based and item-based collaborative filtering (in Python, using surprise). Include a simple popularity/mean baseline as a benchmark. Evaluate performance using RMSE for rating prediction and Precision@K / Recall@K for Top-N recommendations. Which model performs better and why? If your collaborative-filtering models do not outperform the baseline, discuss why (e.g., data sparsity, the baseline being a strong "popularity" recommender) and what that implies for model selection.
Add an AI re-ranking layer on top of your recommender (Refer to Week 4 materials). Take the Top-N candidates from your best collaborative-filtering model and use an LLM to re-rank and personalize them to a user’s stated preference (e.g., a mood, favorite genre, or other preferences), returning a short explanation for each pick. The LLM should re-rank the candidate recommendations produced by your collaborative-filtering model rather than generate entirely new recommendations using the book metadata (e.g., author, title, year, average rating, ...) as context. You may experiment with different prompt styles or personalization strategies. You may use any LLM (cloud or local) for this layer - Gemini is recommended because it's free (see the API-key guide) - but cite the model and provider you used, and never include your key in the submission.

Build a Streamlit app that wraps your recommender. (Refer to Week 3 and 4 demos and class materials and the Guides section below on how to build and run a Streamlit app). The app should let a user be selected, show the collaborative-filtering Top-N, and offer the LLM re-ranking.

Record a short (~2–3 minute) video demo walking through your app and briefly explaining how it works - the CF step, the LLM step, and one design choice you made. Do not include your API key in your code or submission.
Business discussion. What are the business applications of each model (collaborative filtering and the LLM layer) and what challenges might a business face when setting these up? What would be your recommended approach for this dataset?
 

Deliverables
A set of slides (PDF format). In the slide deck, present your two system instruction prompts and your two judges, and summarize the score comparison between Prompt A and Prompt B. Include an interpretation of where and why the prompts differed and a discussion of selected Braintrust traces. Add a slide on the evaluation design itself: its limitations, how you would improve it, and what other types of evaluation are relevant to a recommendation and re-ranking problem beyond a single LLM judge. Include your Braintrust evidence as exhibits in the deck, the results comparison (using the Experiment page's "Summary Table" layout), and the selected traces, using the appendix slides if needed. The deck should include a title slide with the team number and names, use a font size of 16 or higher, and not exceed 8 slides, including the title slide (you may include up to 2 appendix slides for supporting exhibits). Submit the PDF version of the slides.
 

Guides
...
 

Rubric:

Your project will be evaluated on:

...
 

Grading:

While team members are expected to receive the same grade, the instructor reserves the right to adjust each individual’s grade or assign 0 for the individual’s project grade in the event that a student does not actively participate in the team’s activities and does not contribute to the team deliverables.

 

Rubric
Project Two
Project Two
Criteria	Ratings	Pts
This criterion is linked to a Learning OutcomeExploratory Data Analysis & Insights
Insightful exploration of users, ratings, and books; patterns/anomalies that could affect modeling or recommendations.
10 pts
This criterion is linked to a Learning OutcomeRecommendation Modeling & Evaluation
Appropriate implementation and evaluation of user-based and item-based collaborative filtering with a baseline; correct hold-out evaluation (RMSE, Precision@N / Recall@N); model comparison, interpretation of model behavior, and justified recommendation approach.
30 pts
This criterion is linked to a Learning OutcomeLLM Personalization Layer
Sound prompt design and structured output; a re-ranking step that meaningfully personalizes the recommendations to a stated preference, with clear, relevant explanations.
20 pts
This criterion is linked to a Learning OutcomeStreamlit App & Video Demo
A working app that integrates the recommender and the LLM layer; a clear ~2–3 minute video demo that explains how it works.
15 pts
This criterion is linked to a Learning OutcomeSlides & Communication
Clear, professional, well-organized deck - methodology, findings, conclusions, and lessons learned are easy to follow.
15 pts
This criterion is linked to a Learning OutcomeCode Quality & Reproducibility
Organized, annotated code that reproduces by reading Books.csv / Ratings.csv (no API key included).
5 pts
This criterion is linked to a Learning OutcomeCreative Design & Extra Features
Thoughtful UX, additional features, or work that goes beyond the base requirements.
5 pts
Total Points: 100