"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Vectorizer
news_vectorizer = open("resources/vector.pickel","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# def vec(text):
# #using countVectoerizer
# vectorizer = CountVectorizer(lowercase=True, stop_words='english', analyzer='word', ngram_range=(1, 1))


# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "EDA", "Team"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.title("This is a Tweet Classifer")
		st.subheader("tweet classification")
		image = Image.open('resources/imgs/ghg.png')
		st.image(image, caption='Climate Change')
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.snow()
		
		st.markdown("Global warming is a phenomenon of climate change characterized by a general increase in average temperatures of the Earth, which modifies the weather balances and ecosystems for a long time. It is directly linked to the increase of greenhouse gases in our atmosphere, worsening the greenhouse effect.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "EDA page" page
	if selection == "EDA":
		st.title("Tweet Classifery EDA")
		image = Image.open('resources/imgs/eda.gif')
		st.image(image, caption='Data Exploration')
		# You can read a markdown file from supporting resources folder
		st.markdown("In this page we will walk you through the exploratory Data Analysis done for the tweeter dataset")
		st.text("Here is the first few rows")
		st.dataframe(raw.head())
		st.markdown("After taking a look at the frist five  rows of the DataFrame we notice that we have `Three(3)` columns in the dataFrame. we have two features and one label,  **features** inludes: - `message` - `tweetid`,  **label**: - `sentiment` And the test dataFrame contains only the features")
		# taking a closer look at the tweets
		st.markdown("Taking a closer look at the `message` column, it appears they alot of emojis and other unwanted symbols")
		st.write(raw['message'][3195])
		# taking a look at the labels
		image = Image.open('resources/imgs/word_cloud.png')
		st.image(image, caption='Word Cloud')
		st.markdown("Here is label in our dataset")
		st.write(raw['sentiment'].value_counts())
		col1, col2, col3, col4 = st.columns(4)
		col1.metric("PRO tweets '1' ", "52.25 %", "Supports")
		col2.metric("Nuetral tweets '0' ", "17.55 %",
		            "Neither supports nor refutes", delta_color="off")
		col3.metric("NPOR  '-1' ", "9.08 %", "Do not believe", delta_color="inverse")
		col4.metric("News '2' ", "21.10 %", "Climate change news")
		st.markdown("A countplot of the labels")
		def countplot():
			fig = plt.figure(figsize=(10, 4))
			sns.countplot(x = "sentiment", data = raw)
			st.pyplot(fig)
		countplot()
		st.markdown("The labels appears to be Umbalanced")

	# Building the team page
	if selection == "Team":
		st.title("The Team")	
		st.markdown("Here is the Awesome team behind this project")
		image = Image.open('resources/imgs/profile_pic.png')
		st.image(image, caption='Team lead Abubakar',  width=300)
		col1, col2, col3 = st.columns(3)
		with col1:
			image = Image.open('resources/imgs/raf.jpg')
			st.header("Team members")
			st.image(image, caption= 'Raphael', width=300)
		with col2:
			image = Image.open("resources/imgs/Dienebi.jpg")
			st.image(image, caption='Dienebi', width=300)
		with col3:
			image = Image.open("resources/imgs/Mijan.jpg")
			st.image(image, caption='Mijan', width=300)
		with col3:
			image = Image.open("resources/imgs/Ahamd.jpg")
			st.image(image, caption='Ahmad', width=300)
		st.balloons()
	# Building out the predication page
	if selection == "Prediction":
		st.title("This is a Tweet Classifer")
		st.subheader("tweet classification")
		image = Image.open('resources/imgs/climate_pic.jpg')
		st.image(image, caption='Climate Change')
		st.info("Predicting tweets using ML Models")
		option = st.selectbox(
                    'Select the model of your choose from the drop down',
                    ('Logistic Regression', 'SVC', 'RandomForest'))
		st.write('You selected:', option)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		# model selection options
		if option == 'Logistic Regression':
			model = "resources/logistic_model.pkl"
		elif option == 'SVC':
			model = "resources/svc_model.pkl"
		elif option == 'RandomForest':
			model = "resources/forest_model.pkl"

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)
			
			word = ''
			if prediction == 0:
				word = '"**Neutral**". It neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				word = '"**Pro**". The tweet supports the belief of man-made climate change'
			elif prediction == 2:
				word = '**News**. The tweet links to factual news about climate change'
			else:
				word = 'The tweet do not belief in man-made climate change'
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as {}".format(word))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
