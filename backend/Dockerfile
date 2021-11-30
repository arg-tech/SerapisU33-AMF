FROM python:3.6.8
COPY . /app
WORKDIR /app

RUN pip install --upgrade pip

RUN pip3 install tqdm
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install torch
RUN pip3 install numpy
RUN pip3 install transformers
RUN pip3  install nltk 
RUN pip3  install sklearn 
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader averaged_perceptron_tagger



RUN pip3 install spacy

RUN pip3 install Cython
RUN python -m spacy download en
RUN python -m spacy download en_core_web_lg
RUN pip3 install sacrebleu
RUN pip3 install -U Werkzeug






RUN pip install -r requirements.txt
EXPOSE 5009
CMD python ./main.py