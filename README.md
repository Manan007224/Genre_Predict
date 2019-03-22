# Genre Predict

## Objectives

1) To prove that a movies plot summary has no relation whatsover with the critic ratings or there is no such 'favourite' genres that critics like

2) To predict a movie's genres using plot summary with the help of nltk packages

## Technical Details

The code is written in Python 3.6 version. To setup the environment with all required dependencies you must have first jupyter-notebook via terminal or using Anaconda cloud. Also pip3 must be installed in the command line to install the packages

```javascript
pip3 install pandas
pip3 install matplotlib
pip3 install sklearn
pip3 install seaborn
pip3 install nltk
pip3 install spacy
python3 -m spacy download en_core_web_sm
```

## Usage

To execute the python3 code, clone this repository and go inside this directory and write:
```javacript
python3 nlp.py
```

## Reading Output

The code outputs model scores, but further more exploration you can load the .ipynb file which is inside Jupyter-Notebook called kernel.ipynb for easier access.

## Additional Information
Problem Addressed:
The goal of this assignemnt is to prove that plot summary given by IMDB had no relation with Rotten Tomatoes critic ratings.

Getting/Cleaning Data:
Data is taken from IMDB and Rotten Totatoes. Textual information is provided by IMDB and information like critic rating, audience rating etc is provided by Rotten Tomatoes. In the program, the 2 datasets are merged and uncenessary colums are removed. The obvious intuition was to apply regression methods to plot summary and critic percent column and state that these aren't correlated, but the problem was to convert the textual data (plot summary) into some form of numerical data. Program first tokenizes the summary (separate into words), removes the punctuations (',', '.', '!'. '%') and stop words ('is', 'a', 'the', 'and'). After that only nouns and verbs are left in the plot summary column. Some separate columns like purity (Weighted Words/Total word count) are added to the data. Also, some of the columns and rows (where the values were Nan) are removeed to avoid redundant data. After the data cleaning, around 30% of the data points are lost but at least it was not redundant. Also some rearrangements are done in the dataset’s columns in order to get according to the needs.

Data Analysis:
After cleaning the data, I had to apply TFIDF vectorize to my plot summary data to reflect how important a word is to a document in a collection or corpus. Tfidf is just a simple ranking algorithm which works on term frequency. After applying to the TFIDF I was left with a N*N matrix which represented the plot summary in numerical form. N is the no. of distinct words in the plot summary column (after cleaning, as having stop words like 'a', 'is' make no sense). I tested my training
  
Results:
LassoRegression(Alpha=0.1) = 0.0813 
LassoRegression(Alpha=0.2) = 0.0093 
LassoRegression(Alpha=0.6) = 0.0147 
LassoRegression(Alpha=0.75) = 0.0267
The graph shows how the Lasso Regression performed over different range of alpha values.

Limitations:
Initially I wanted to make a recommendation engine which predicts the top 3 related genres to an input of a plot_summary. But in the dataset for a plot_summary there was a list of genres and for throwing the training sets into classification Algorithms I had to have just one label(genre) for each plot_summary but wasn't able to find some sensible method to concatenate 3 genres per plot into one.
