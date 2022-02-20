FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root"]