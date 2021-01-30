# Introduction

A wide range of transformer-based models started coming up for different NLP tasks. There are multiple advantages of using transformer-based models, but the most important ones are:

## First Benefit

- These models do not process an input sequence token by token rather they take the entire sequence as input in one go which is a big improvement over RNN based models because now the model can be accelerated by the GPUs.

## Second Benefit

- We don’t need labeled data to pre-train these models. It means that we have to just provide a huge amount of unlabeled text data to train a transformer-based model. We can use this trained model for other NLP tasks like text classification, named entity recognition, text generation, etc. This is how transfer learning works in NLP.

BERT and GPT-2 are the most popular transformer-based models and in this article, we will focus on BERT and learn how we can use a pre-trained BERT model to perform text classification.

**Reference:**

- [Analyticsvidya Blog - starter code](https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/)
- [bert-sentiment - Abhishek Thakur](https://github.com/abhishekkrthakur/bert-sentiment/blob/master/train.py)


# Why finetune language model:

We often have large quantity of unlabelled dataset with only a small amount of labelled dataset.If we need to get accurate classification, we can use pretrained models trained on large corpus to get decent results. Generally, we use pretrained language models trained on large corpus to get embeddings and then mostly add a layer or two of neural networks on top to fit to our task in hand. This works very well until the data on which language model was trained is similar to our data. 

**Problem:** If our data is different than data used for pretraining, results would not be that satifactory. Consider for example if we have mix of Hindi and English language data and we are using pretrained model trained on Wikipedia, it would lead to bad results. 

**Solution:** In that scenario we need to fine-tune our language model too. As shown by Jeremy Howard and Sebastian Ruder in [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146), finetuning the language model can lead to performance enhancement. We generally modify the last few layers of language models to adapt to our data. This has been done and explained by Fast.ai in [Finetuning FastAI language model](https://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html). They have done it extensively for `ULMFit`. We can follow the same approach with Bert and other models. With the revolution in NLP world, and with the arrival of beasts such as Bert, OpenAI-GPT, Elmo and so on we need a library which could help us keep up with this growing pace in NLP. Here comes in Hugging Face pytorch-transformers, a one stop for NLP. This is easy to use library to meet all your NLP requirements written in Pytorch. We will see how we can fine-tune Bert language model and then use that for SequenceClassification all using pytorch-transformers.
