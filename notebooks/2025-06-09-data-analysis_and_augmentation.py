# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploration of user feedback data

# %% [markdown]
# This notebook contains analysis performed as part of technical exercise for the position of Data Scientist with GDS Product Group.
#
# Notebook structure:
# * Section 1: Investigating the dataset
# * Section 2: Data augmentation
# * Section 3: Detecting themes in user feedback

# %%
import time

import altair
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from theme_explorer.llm.llm_synthetic import SyntheticSurvey
from theme_explorer.utils.text_cleaning import clean_text, flag_spam_content

# %% [markdown]
# ## Section 1: Investigating the dataset

# %% [markdown]
# ### Load provided dataset

# %%
data = pd.read_csv("../data/raw/feedback_data.csv")

# %%
data.describe()

# %% [markdown]
# Quick look at the dataset shows that it contains two variables: `Feedback` and `Sentiment`. The `Feedback` is typically is a short sentence and the `Sentiment` is a binary label.
#
# It also appears that some `Feedback` values are duplicates.

# %%
# Look at a few randomly selected examples
data.sample(3).T

# %%
# Check Sentiment distribution
data["Sentiment"].value_counts()


# %% [markdown]
# In the provided fictional dataset positive and negative sentiment labels are represented fairly equally.

# %%
# Check distribution of feedback length
data["feedback_length"] = data["Feedback"].str.len()

# Plot the distribution of feedback lengths broken down by sentiment type
sns.histplot(data=data, x="feedback_length", hue="Sentiment", multiple="stack")

# %%
print(f"The median length of feedback is {data['feedback_length'].median()} ")
print(
    f"For negative feedback this values is {data[data['Sentiment'] == 'Negative']['feedback_length'].median()} characters"
)
print(
    f"For positive feedback the median length of feedback is {data[data['Sentiment'] == 'Positive']['feedback_length'].median()} characters"
)

# %% [markdown]
# Positive feedback appears to be longer than negative. At the same time there are outliers for negative feedback that are noticeably longer.

# %% [markdown]
# ## Section 2: Data augmentation

# %% [markdown]
# ### Data pre-processing

# %% [markdown]
# From initial exploration of the dataset I identified some issues that make it necessary to clean and pre-process the data:
# * Duplicated feedback entries. These entries need to be removed so that they do not skew sentiment distribution.
# * Presence of potentially personally identifiable information (PII) in the form of phone numbers and emails. PII needs to be masked to avoid disclosure and protect users' privacy.
# * Presence of irrelevant entries, such as spam. These entries need to be filtered out as they do not represent user feedback.
#
# To address these issues, I have taken the following pre-processing steps:
# * Removed duplicates
# * Detected and masked potential PII based on patterns (special characters that indicate emails, series of numeric characters resembling phone numbers, capitalisation patterns that might refer to names)
# * Flagged potential spam based on patterns (presence of urls, atypical capitalisation)
#
# Also, to address potential issues with data quality coming from web sources, I have also removed punctuation, standardised whitespaces, checked for encoding issues and converted text to lower case.
#
# To ensure readability of the notebook and in line with best practice of writing modular code, I have put most of the pre-processing code into the `text_cleaning` module. The module is included in the `theme_explorer` git repo folder attached to this submission.

# %%
data["clean_feedback"] = clean_text(data["Feedback"])

# %%
data["spam_flag"] = flag_spam_content(data["Feedback"])

# %%
# Check known spam entry
data.loc[48]

# %%
# Filter out spam entries and duplicates
data = data[~data["spam_flag"]]
data = data.drop_duplicates(subset=["Feedback"])

# %% [markdown]
# ### Data augmentation
#
# For data augmentation I used Claude Large Language Model (LLM) to programmatically generate synthetic data using an API.
#
# This involved:
# * Defining prompt template (see `src/theme_explorer/llm/prompt_synthetic.py`). This enables editing prompts in a systematic way.
# * Specifying requirements for the LLM output using Pydantic (see `src/theme_explorer/llm/synthetic_response.py`). Having a data model for the LLM response helps to ensure that LLM output is returned in a structured format for subsequent processing.
# * Defining a function to prepare and send the prompt and additional information to LLM (see `src/theme_explorer/llm/llm_synthetic.py`)
#
# LLM prompt includes pairs of feedback with negative and positive sentiment. This is done as part of few shot prompting to make LLM-generated data more realistic.
#

# %%
concept_definitions = """'Feedback' is a response from the user of a web-site or an app describing their experience.
It contains on average one short sentence (70 characters).
'Sentiment' is a label referring to the sentiment of the feedback. It can be Negative or Positive'. """

# %%
# List of positive examples as a dict
positive_examples = data[data["Sentiment"] == "Positive"].to_dict(orient="records")
negative_examples = data[data["Sentiment"] == "Negative"].to_dict(orient="records")

# %%
survey = SyntheticSurvey(model_name="claude-sonnet-4-20250514")

example_pairs = list(zip(positive_examples, negative_examples))

response_dfs = []
for i, pair in enumerate(example_pairs):
    responses = survey.get_synthetic_response(
        concept1="user feedback",
        concept2="sentiment",
        definitions=concept_definitions,
        examples=pair,
        instructions="Generate 10 new user feedback responses with sentiment labels",
    )

    time.sleep(10)  # Wait to avoid rate limiting

    # Convert response to dictionary and create a DataFrame
    response_data = [
        {"feedback": item.feedback, "sentiment": item.sentiment}
        for item in responses.responses
    ]
    response_dfs.append(pd.DataFrame(response_data))


# %%
# Now responses is a SyntheticResponse object
print(f"Here are {len(responses.responses)} sample feedback items")

# Access individual feedback items
for item in responses.responses[:5]:
    print(f"Feedback: {item.feedback}")
    print(f"Sentiment: {item.sentiment}")
    print("---")

# %%
synthetic_df = pd.concat(response_dfs)

# %%
synthetic_df.describe()

# %% [markdown]
# There are a few duplicated entries in the synthetic dataset.

# %%
synthetic_df["sentiment"].value_counts()

# %% [markdown]
# Synthetic data maintains balance between negative and positive sentiment labels.

# %%
# Check distribution of feedback length
synthetic_df["feedback_length"] = synthetic_df["feedback"].str.len()

# Plot the distribution of feedback lengths broken down by sentiment type
sns.histplot(data=synthetic_df, x="feedback_length", hue="sentiment", multiple="stack")

# %%
print(
    f"The median length of synthetic feedback is {synthetic_df['feedback_length'].median()} "
)
print(
    f"For negative feedback this values is {synthetic_df[synthetic_df['sentiment'] == 'Negative']['feedback_length'].median()} characters"
)
print(
    f"For positive feedback the median length of feedback is {synthetic_df[synthetic_df['sentiment'] == 'Positive']['feedback_length'].median()} characters"
)

# %% [markdown]
# While synthetic data has a similar distribution of feedback length across sentiment, more work is needed to check that synthetic data is not materially different from the original dataset.

# %%
synthetic_df.to_csv("../outputs/synthetic_feedback_data_2025-06-14.csv", index=False)

# %% [markdown]
# ## Section 3: Detecting themes in user feedback
#
# Once we have accumulated a larger dataset of user feedback, we can start exploring possible themes.
#
# Below are some examples of analysis that we could perform for this purpose.

# %% [markdown]
# ### Example 1: Identifying top terms across negative and positive feedback
#
# In this example I use very simple counts of common terms across negative and positive feedback.

# %%
data.rename(columns={"Feedback": "feedback", "Sentiment": "sentiment"}, inplace=True)

# %%
synthetic_df["clean_feedback"] = clean_text(synthetic_df["feedback"])

# %%
# Combine original data and synthetic data
combined_data = pd.concat(
    [data[["feedback", "sentiment", "clean_feedback"]], synthetic_df], ignore_index=True
)

# %%
combined_data = combined_data.drop_duplicates(subset=["feedback"])

# %%
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")
counts = vectorizer.fit(combined_data["clean_feedback"])

# %%
# Identify most frequent ngrams in counts
positive_feedback = combined_data[combined_data["sentiment"] == "Positive"]
negative_feedback = combined_data[combined_data["sentiment"] == "Negative"]
top_ngrams_pos = (
    pd.DataFrame(
        counts.transform(positive_feedback["clean_feedback"]).toarray(),
        columns=counts.get_feature_names_out(),
    )
    .sum()
    .sort_values(ascending=False)
    .head(25)
)
top_ngrams_neg = (
    pd.DataFrame(
        counts.transform(negative_feedback["clean_feedback"]).toarray(),
        columns=counts.get_feature_names_out(),
    )
    .sum()
    .sort_values(ascending=False)
    .head(25)
)

# %%
# Plot most popular 25 ngrams in positive feedback
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.barplot(x=top_ngrams_pos.values, y=top_ngrams_pos.index)
ax.set_title("Top 25 ngrams in positive feedback")
plt.savefig("../outputs/top_ngrams_pos.png")

# %% [markdown]
# Reviewing top terms in positive feedback can help us understand aspects of user experience that work well (e.g. 'search', 'design`).

# %%
# Plot most popular 25 ngrams in negative feedback
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.barplot(x=top_ngrams_neg.values, y=top_ngrams_neg.index)
ax.set_title("Top 25 ngrams in negative feedback")
plt.savefig("../outputs/top_ngrams_neg.png")

# %% [markdown]
# Similarly, looking at top terms in negative feedback enables us to spot potential issues (e.g. 'navigation menu', 'text small')

# %% [markdown]
# ### Example 2: Cluster feedback
#
# In this example, we embed feedback sentences using `sentence-transformers` package, cluster them using HDBSCAN method and then visualise clusters using UMAP 2-d projections.

# %%
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# %%
feedback_embeddings = model.encode(
    list(combined_data["clean_feedback"]),
)

reducer = umap.UMAP(n_components=2, random_state=1)
feedback_embeddings_2d = reducer.fit_transform(feedback_embeddings)

# %%
# Low dim embeddings
reducer_clustering = umap.UMAP(n_components=20, random_state=1)
feedback_embeddings_clustering = reducer_clustering.fit_transform(feedback_embeddings)

# %%
# Df for altair plotting
col_df = pd.DataFrame(
    {
        "feedback": list(combined_data["clean_feedback"]),
        "x": feedback_embeddings_2d[:, 0],
        "y": feedback_embeddings_2d[:, 1],
    }
)

# %%
# Cluster with hdbscan
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2, min_samples=2, cluster_selection_method="leaf"
)
clusterer.fit(feedback_embeddings_clustering)

# %%
col_df["cluster"] = [str(x) for x in clusterer.labels_]

# %%
fig = (
    altair.Chart(col_df, width=800, height=800, title="Feedback clusters")
    .mark_circle(size=60)
    .encode(
        x="x",
        y="y",
        tooltip=["feedback", "cluster"],
        color=altair.Color("cluster", scale=altair.Scale(scheme="category20")),
    )
).interactive()

fig.configure_title(fontSize=24)

# %%
fig.save("../outputs/feedback_clusters_2025-06-14.html")

# %% [markdown]
# Clustering feedback and profiling resulting clusters can help us detect themes we may not be aware of. It can also serve for validation purposes to compare to outputs from another method, such as LLM theme detection and summarisation.
