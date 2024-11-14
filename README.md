# Titanic 생존 예측 모델

이 프로젝트는 Titanic 생존 데이터를 사용하여 탑승객의 **Pclass(티켓 클래스)**와 **Age(나이)**가 생존율에 미치는 영향을 분석하고 예측하는 머신러닝 모델을 구축하는 것을 목표로 합니다. 이 프로젝트는 Python을 기반으로 하고 있으며, 관련 데이터 전처리와 모델 훈련을 위한 코드와 설명이 포함되어 있습니다.

## 데이터셋 링크
https://www.kaggle.com/c/titanic/data

## nbviewer 링크
https://nbviewer.org/github/YeonJoon-Ryu/MLproject/blob/main/ryu_tatanic.ipynb

## 프로젝트 개요

Titanic 데이터셋은 여러 요인들이 생존 확률에 영향을 미쳤던 상황을 바탕으로 만들어졌습니다. 이 모델은 **Pclass**와 **Age**의 상관관계가 생존 여부에 미치는 영향을 예측하는 것에 초점을 맞춥니다.

## 환경 설정

프로젝트에서 사용된 주요 라이브러리 및 버전은 다음과 같습니다.

- **Python**: 3.10.11
- **NumPy**: 1.24.3
- **Pandas**: 1.5.3
- **Seaborn**: 0.12.2
- **Scikit-Learn (sklearn)**: 1.2.2

## 데이터 전처리 및 피처 선택

본 프로젝트는 다음과 같은 Feature를 선택하여 모델을 훈련합니다.

- **Pclass**: 객석 등급 (1, 2, 3등석)
- **Age**: 나이

위 두 가지 피처가 생존율과의 관계가 있는지를 분석하고, 머신러닝 모델을 통해 예측하는 것이 주요 목표입니다.

## 설치 가이드

1. **Poetry 설치**  
   프로젝트를 설치하려면 Poetry가 필요합니다. Poetry가 설치되어 있지 않다면, 다음 명령어를 통해 설치하십시오:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -

---

# Titanic Survival Prediction Model

This project aims to build a machine learning model using Titanic survival data to analyze and predict the impact of **Pclass (Ticket Class)** and **Age** on survival rates. The project is based on Python and includes code and explanations for data preprocessing and model training.

## Project Overview

The Titanic dataset was created based on various factors that influenced survival probabilities. This model focuses on predicting the effect of the relationship between **Pclass** and **Age** on survival outcomes.

## Environment Setup

The main libraries and versions used in this project are as follows:

- **Python**: 3.10.11
- **NumPy**: 1.24.3
- **Pandas**: 1.5.3
- **Seaborn**: 0.12.2
- **Scikit-Learn (sklearn)**: 1.2.2

## Data Preprocessing and Feature Selection

This project selects the following features to train the model:

- **Pclass**: Passenger class (1st, 2nd, 3rd class)
- **Age**: Age

The main goal is to analyze the relationship between these two features and survival rates, using a machine learning model to predict any significant correlation.

## Installation Guide

1. **Install Poetry**  
   To install the project, Poetry is required. If Poetry is not already installed, install it using the following command:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
