# A Reflection by an AI Therapist

In this challenge we trained a [distilled](https://arxiv.org/abs/1910.01108) generative neural network based on the [transformer architecture](https://arxiv.org/abs/1706.03762) ([distilgpt2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) in order to generate short, contextualized, and empathetic reflections.

The general idea of the model is simple, can we learn a mapping from patient cases to therapist reflections?  The short answer is yes, but there are also many obstacles to conquer, which we will discuss later in this document. 

For training and testing the models, we suggest using the notebooks in this repository on [Google Colab](https://colab.research.google.com/) as it provides free GPU (which the models need) and also most packages we need are already provided in their environment. Now, let’s start by going over different notebooks that we have in this repository.


## Notebooks


### Data Preparation

The data preparation is performed in `data_preparation.ipynb` which takes the prepared dataset and returns a single `.txt` file. Our dataset will be a text file were each entry is a patient's case followed by the following sentences depending on their availability:

 1. the first sentence of the therapist's massage.

 2. the sentence in the which either of these keywords appear:

   * seems like

   * sounds like

   * feels like


### Model Training

Using the dataset we prepared, we trained a model for 2.1k steps in `model_training.ipynb` notebook. The training objective of the function was set to Casual Language Modeling objective which directly correlates with the perplexity of the model. In these settings, perplexity is a measure of how good the algorithm is able to understand the language and model it effectively. Perplexity measures how confused the model is when it is trying to generate a new sequence, so the lower it is the better. Since our is a function of perplexity, we can just look at the loss function to know how well the model is doing.

We divided our dataset to 95% training and 5% validation data. The result of our evaluation for these two datasets is given below:

#### Training Loss



#### Validation Loss



Once the training process is over the model is saved in the `fine_tune` directory and it is ready to be picked up for generation.


### Sequence Generation

The `sequence_generation.ipynb` notebook, loads a saved model and upon receiving a patient case, it will generate `n` possible (probable) responses that the therapist might say based on our training data. In the script we set `n=3` but this is a changeable parameter and we can use it to generate any number of responses we would want for a single patient case.


# Summary

The most challenging part of this project (aside from long hours of coding) was the data. I spent a significant amount of time searching for sources of data that could augment the training data for better and smoother results, however I was not successful in finding external datasets. However, I believe we could find supplementary data with more time.

The other challenging part about this project was my limited experience in psychology, which prevented me from being able to mine for more data from the provided dataset. I followed the tips that were on the slides and they worked well. However the model is far from perfect, and we need more data to have a robust model.

Since the model is based on [attention mechanism ](https://arxiv.org/abs/1409.0473)we can visualize what the model is paying attention to during the suggestion time. Which gives us a gateway to explain the model behavior for analysis and improvement. 

Next, we should think about the efficiency of the model in memory and computation, which I have implemented a way to optimize for it but further experiments are needed to see which optimization method is best for our use case.

Other future improvements of this module would be equipping it with multilingual understanding of the user, which expands the resources we can use to train a single model while providing us with better generative behaviors and coverage of other spoken languages.