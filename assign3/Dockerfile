FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
CMD ["ccocca2_assign3.py"]
ENTRYPOINT ["python"]