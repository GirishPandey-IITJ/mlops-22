FROM ubuntu:latest
#FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./api/flask_program_56.py /exp/api/flask_program_56.py
COPY ./api/dt_max_depth=20.joblib /exp/api/dt_max_depth=20.joblib
RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
# CMD ["python3", "./plot_graph.py"]
CMD ["python3","./api/flask_program_56.py"]