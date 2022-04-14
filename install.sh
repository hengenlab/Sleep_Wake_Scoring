# conda create -n Sleep_Wake_Scoring python=3.7
conda install numpy scipy joblib pandas matplotlib ipython opencv seaborn\
              scikit-learn psutil h5py pillow

pip install opencv-python
pip install -U scikit-learn==0.21.3

pip install 'neuraltoolkit@git+https://github.com/hengenlab/neuraltoolkit.git'
pip install 'musclebeachtools@git+https://github.com/hengenlab/musclebeachtools_hlab.git'
