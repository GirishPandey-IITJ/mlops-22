FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./plot_digits_classification.py /exp/plot_digits_classification.py 
COPY ./svm_gamma=0.001_C=1.5.joblib /exp/svm_gamma=0.001_C=1.5.joblib
RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
ENV CLF_NAME = "svm"
ENV RANDOM_STATE = 42
CMD python plot_digits_classification.py ${CLF_NAME} ${RANDOM_STATE}
