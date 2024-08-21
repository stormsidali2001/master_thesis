# LEVERAGING BERT AND DATA AUGMENTATION FOR ROBUST CLASSIFICATION OF IMRAD SECTIONS IN RESEARCH PAPERS

## Introduction

This repository contains the code and Jupyter notebooks for my Master's thesis research, "LEVERAGING BERT AND DATA AUGMENTATION FOR ROBUST CLASSIFICATION OF IMRAD SECTIONS IN RESEARCH PAPERS". The thesis focuses on developing a machine learning model to accurately classify sentences within research papers into their respective IMRAD (Introduction, Methods, Results, and Discussion) sections. This classification task is crucial for various applications, including information retrieval, automated summarization, and content recommendation systems. 

This project utilizes advanced techniques like BERT (Bidirectional Encoder Representations from Transformers) and data augmentation to improve the model's accuracy and generalization capabilities.

## Repository Structure

The core components of the research are organized within the `/notebooks_pipeline` directory. The notebooks follow a logical workflow and are designed to be executed sequentially:

1. **[1.preparing_data.ipynb](/notebooks_pipeline/1.preparing_data.ipynb):**  Covers the initial data acquisition, preprocessing, and preparation of the `unarXive_imrad_clf` dataset sourced from Hugging Face. 
2. **[2.first_bert_training.ipynb](/notebooks_pipeline/2.first_bert_training.ipynb):**  Details the first iteration of training a BERT model on the preprocessed dataset. This notebook likely includes hyperparameter tuning and initial performance evaluation.
3. **[3.outlier_detection_data_cleaning.ipynb](/notebooks_pipeline/3.outlier_detection_data_cleaning.ipynb):**  Focuses on identifying and handling outliers in the dataset, which can negatively impact model performance. The notebook likely utilizes a language model (possibly 'gemini-1.5-flash') for classification and outlier detection.
4. **[4.master_july_30_bert_training_after_gemini_cleaning.ipynb](/notebooks_pipeline/4.master_july_30_bert_training_after_gemini_cleaning.ipynb):**  Documents a refined BERT training process, incorporating insights and the cleaned data resulting from the outlier detection phase.
5. **[5.data_augmentation.ipynb](/notebooks_pipeline/5.data_augmentation.ipynb):**  Explores data augmentation strategies to increase the size and diversity of the training data, using the 'gemini-1.5-flash' model to generate synthetic but representative text samples.
6. **[6.master_16_augest_bert_training_after_gemini_cleaning.ipynb](/notebooks_pipeline/6.master_16_augest_bert_training_after_gemini_cleaning.ipynb):**  Presents the final BERT model training iteration, leveraging the augmented dataset and likely incorporating further model fine-tuning. 


## Results and Discussion

The research demonstrated the effectiveness of BERT and data augmentation for IMRAD section classification. The final BERT model achieved an F1 score of 0.9172 on the test set, a substantial improvement over the baseline Logistic Regression model. The combination of transfer learning and data augmentation techniques significantly improved model robustness, accuracy, and generalization to unseen data.



## Contact

Sid Ali Assoul - [assoulsidali.contact@gmail.com] 

