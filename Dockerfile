FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
COPY TACO-master TACO
RUN pip install pillow requests numpy pandas matplotlib seaborn graphviz Cython