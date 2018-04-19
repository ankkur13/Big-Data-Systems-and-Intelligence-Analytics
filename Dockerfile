# Use the basic Python 3 image as launching point
FROM python:3.6.3

# Add the script or text to the Dockerfile
ADD model_deployment_script.py /home
# ADD requirements.txt /home
ADD Scrapped_content.csv /home
ADD argv_input_syntax.txt /home

# Install required Libraries
# RUN pip install -r ./home/requirements.txt
RUN pip install numpy
RUN pip install pandas
RUN pip install seaborn
RUN pip install sklearn
RUN pip install scipy  
RUN pip install sklearn  
RUN pip install nltk
#RUN pip install string
#RUN pip install re
RUN pip install stop_words  
#RUN pip install collections
RUN pip install wordcloud
RUN pip install textblob
RUN pip install xgboost
#RUN pip install PIL
#RUN pip install pickle
RUN pip install boto
#RUN pip install sys



#CMD ["python3", "./home/model_deployment_script.py"]