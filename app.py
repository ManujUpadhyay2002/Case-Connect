from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import streamlit as st
import pandas as pd
import spacy

st.title(':violet[Case  :orange[Connect]]')


class App:
    def __init__(self):
        """
        Initializes the App class.
        Loads the pre-trained language model and the DataFrame containing judgment vectors.
        """
        self.nlp = spacy.load("en_core_web_md")
        self.df = pd.read_pickle('vector_0_100.pkl')

    def userInput(self):
        """
        Processes user input and returns relevant judgments.

        Returns:
            DataFrame: A DataFrame containing the top relevant judgments based on the user input.
                       The DataFrame has columns for petitioner name, respondent name, and judgment links.
            str: An empty string if the user input is empty.
        """
        user_input = st.text_area('Enter Case: ', '''''')
        if user_input != '' and detect(user_input) == 'en':
            vector = self.textProcessing(user_input)
            locations = self.findJudjement(vector)
            df = pd.DataFrame(
                columns=['Petitioner Name', 'Respondent Name', 'Judgement Links'])
            for index, location in enumerate(locations):
                pName = self.df.iloc[location[0]]['petitionerName']
                rName = self.df.iloc[location[0]]['respondentName']
                link = self.df.iloc[location[0]]['judgmentLink'][location[1]]
                df.at[index, 'Petitioner Name'] = pName
                df.at[index, 'Respondent Name'] = rName
                df.at[index, 'Judgement Links'] = link
            return df
        else:
            return ''

    def textProcessing(self, user_input):
        """
        Processes the user input and converts it into a vector representation.

        Args:
            user_input (str): The user input to be processed.

        Returns:
            numpy.ndarray: The vector representation of the processed user input.
        """
        doc = self.nlp(user_input)
        cleanedTokens = [token.lemma_.lower()
                         for token in doc if not token.is_punct and not token.is_stop]
        if len(cleanedTokens) != 0:
            vector = self.nlp(" ".join(cleanedTokens)).vector
            return vector
        else:
            return ''

    def findJudjement(self, userVector):
        """
        Finds relevant judgments based on the similarity between the user vector and the judgment vectors.

        Args:
            userVector (numpy.ndarray): The vector representation of the user input.

        Returns:
            list: A list of tuples containing the indices of relevant judgments and their similarity scores.
        """
        if userVector != '':
            tempsimilarity = []
            for index1, courtVectors in enumerate(self.df['vectors']):
                for index2, courtVector in enumerate(courtVectors):
                    tempsimilarity.append((index1, index2, cosine_similarity(
                        userVector.reshape(1, -1), courtVector.reshape(1, -1))[0][0]))
            return sorted(tempsimilarity, key=lambda x: x[2], reverse=True)
        else:
            return ''


test = App()
judgements = test.userInput()
if st.button('Find'):
    st.write(":green[TOP RESULTS:]")
    if type(judgements) == str:
        st.write(
            "--> :orange[First Please Enter Case Detail And Then Press :red[Button]]")
    elif ~judgements.empty:
        for index, row in judgements.iterrows():
            if index < 10:
                st.write(
                    f":orange[{index+1})]  {row['Petitioner Name']} :orange[Vs.] {row['Respondent Name']}")
                st.write(f":orange[Judgement Link] : {row['Judgement Links']}")
            else:
                break
    else:
        st.write(
            "--> :orange[First Please Enter Case Detail And Then Press :red[Button]]")
