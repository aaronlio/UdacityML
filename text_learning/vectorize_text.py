import os
from pickle import TRUE
import joblib
import re
import pickle
import sys
from progressbar import ProgressBar, Percentage, Bar
sys.path.append( "./ud120projects/tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.
    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)
    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.
    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("ud120projects/text_learning/from_sara.txt", "r")
from_chris = open("ud120projects/text_learning/from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

print('[\033[91m LOADING\033[0m ] \033[94m\033[1mEmails are processing right now...\033[0m')

pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=17578).start()
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        # only look at first 200 emails when developing
        # once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        # if temp_counter < 200: #remove comment if you need 200
        path = os.path.join('.', path[:-1])
        # print '[\033[91m OK\033[0m ]' + path
        email = open(path, "r")

        # use parseOutText to extract the text from the opened email
        raw_txt = parseOutText(email)
        # use str.replace() to remove any instances of the words
        # ["sara", "shackleton", "chris", "germani"]
        unwanted_words = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
        for word in unwanted_words:
            raw_txt = raw_txt.replace(word, "") 
        # append the text to word_data
        word_data.append(raw_txt)
        # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        from_data.append(0) if (name == "sara") else from_data.append(1)
        email.close()
        pbar.update(len(word_data))

pbar.finish()

print("Emails Processed")
from_sara.close()
from_chris.close()
print(len(word_data), len(from_data))

pickle.dump( word_data, open("./ud120projects/text_learning/your_word_data.pkl", "wb") )
pickle.dump( from_data, open("./ud120projects/text_learning/your_email_authors.pkl", "wb") )

print(word_data[152])
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
vectorizer.fit_transform(word_data)
print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names()[34597])



