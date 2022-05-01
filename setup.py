from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
   name='Sleep_Wake_Scoring',
   version='0.1.0',
   description='We use a multivariate approach to scoring sleep in high\
                resolution (4 second bins); we use machine learning to\
                adaptively incorporate user input and predict the\
                arousal state of subsequent data. This allows us to\
                score months worth of continuous acquisition in hours.',
   license="",
   long_description=long_description,
   keywords='Sleep_Wake_Scoring, neuroscience, electrophysiology',
   package_dir={'Sleep_Wake_Scoring': 'Sleep_Wake_Scoring'},
   author='\
           (Hengen Lab Washington University in St. Louis)',
   author_email='',
   maintainer='Kiran Bhaskaran-Nair,\
           (Hengen Lab Washington University in St. Louis)',
   maintainer_email='',
   url="https://github.com/hengenlab/Sleep_Wake_Scoring",
   download_url="https://github.com/hengenlab/Sleep_Wake_Scoring",
   packages=['Sleep_Wake_Scoring'],
   install_requires=[
    'ipython', 'numpy', 'matplotlib', 'seaborn', 'pandas',
    'joblib', 'pillow', 'scikit-learn==0.21.3', 'psutil',
    'opencv-python', 'scipy', 'h5py', 'tables',
    'neuraltoolkit@git+https://github.com/hengenlab/neuraltoolkit.git'],
   classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
   scripts=[
           ]
)
