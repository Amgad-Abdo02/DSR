import pandas as pd
import nltk
import string # Used for cleaning punctuation
from gensim.parsing.preprocessing import remove_stopwords #used to remove stop words (unlike nltk it considers however as a stopword)
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup # Used for Removing HTML tags

#Downloading Word datasets used in lemmatization
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')





class Preprocessing:

    def text_preprocessing(self,text:str) -> str:

        """Function Applies Lemmatization and lowercase and removes stop words, Punctiouation, Qoutes 
            and HTML tags 

        Args: data -> (str)

        Returns:
            Cleaned text -> (str) : Preprocessed text for model.

        """
        lemmatizer = WordNetLemmatizer()

        temptxt = BeautifulSoup(text, 'html.parser').get_text()

        temptxt= temptxt.translate(str.maketrans('', '', string.punctuation)) 

        temptxt= temptxt.lower() 

        temptxt= remove_stopwords(temptxt)

        temptxt= temptxt.replace('"', '')

        word_list = word_tokenize(temptxt) 
        clean_words= ' '.join([lemmatizer.lemmatize(w) for w in word_list])

        return clean_words



class Dataset_preprocessing(Preprocessing):

    def Label_preprocessing(self,df: pd.DataFrame, Condition_col_name) -> pd.DataFrame:
        global value_pair
               
        value_pair = {}

        def labeling(condition):
            label=value_pair[condition]
            return label
    
        unique_labels = eval(f"df.{Condition_col_name}.unique()")

        df.condition.value_counts()

 

        for i in range(0,len(unique_labels)):
            value_pair[unique_labels[i]]=i

        df["label"]=df[Condition_col_name].apply(labeling)
        
        

