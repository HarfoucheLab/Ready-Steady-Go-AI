# Ready, Steady, Go AI: A Practical Tutorial on Explainable Artificial Intelligence and Its Applications in Plant Digital Phenomics [![Build Status](https://travis-ci.com/HarfoucheLab/Ready-Steady-Go-AI.svg?token=Hn3qS2qxTmJgJNKTXR7d&branch=main)](https://travis-ci.com/HarfoucheLab/Ready-Steady-Go-AI)
----
![split](http://faridnakhle.com/pv/githubimages/RSGlogo.png?t=1)

This tutorial is a supplement to the paper, **Ready, Steady, Go AI: A Practical Tutorial on Explainable Artificial Intelligence and Its Applications in Plant Digital Phenomics** (submitted to *Patterns, 2021*) by Farid Nakhle and Antoine Harfouche. It aims to introduce the basic principles for implementing artificial intelligence (AI) and explainable AI algorithms in image-based data analysis using the PlantVillage dataset to detect and classify tomato leaf diseases and spider mites as a case study.

Read the accompanying paper [here](https://doi.org) (a link will be available once the paper is published).

Our workflow involves four main steps:
1. Image Dataset Selection
2. Data Preprocessing
3. Data Analysis
4. Performance Analysis and Explanation

As per our workflow, this repository is split into four directories. Each directory contains a description of its corresponding process, and ready to use, interactive notebooks that can run on Google Colab or any other platform supporting Jupyter notebooks.

Click on the directories listed above to proceed.

In the fifth directory (Step 5 - Check Your Understanding, Practice, and Exercise) we have prepared a set of self-test quizzes in form of multiple-choice questions (MCQs), practices, and exercises to provide researchers with opportunities to augment their learning by testing the knowledge they have acquired and applying the concepts explained. Click on the directory for more details.



**UPDATE**

The pretrained DenseNet-161 DCNN algorithm in Data Analysis (step 3) now includes unit tests. 
We are working to complete the unit test coverage for all of our implemented algorithms. 
To run the unit tests, use the following command:
python -m pytest
NB: pytest is required to run the tests. You might install it using pip (pip install pytest).

## Citation
----
If you use any part of this code in your research, kindly cite our paper and our data repository using the bibtex below (bibtex will be updated once the paper is published):

```
@article{
  title={Ready, Steady, Go AI: A Practical Tutorial on Explainable Artificial Intelligence and Its Applications in Plant Digital Phenomics},
  author={Farid Nakhle, Antoine Harfouche},
  journal={},
  pages={},
  year={2021},
  publisher={},
  doi={},
}
```
### Data Repository
http://dx.doi.org/10.17632/4g7k9wptyd.1
(Will be available after publication)