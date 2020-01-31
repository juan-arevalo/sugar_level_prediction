Repositority to explore the potential of Sequence-to-sequence model for blood glucose (BG) prediction.
The raw data is freely available upon email request to <diabetes.artificialintelligence@gmail.com>.


# Intro 

I pose the problem of Blood Glucose (BG) estimation with X hours in advance as a Sequence-to-sequence (seq2seq) auto-regressive model with Recurrent Neural Networks, allowing for attention mechanisim and teacher-forcing. The code consists of two notebooks, prepared to be run in [Google Colab](https://colab.research.google.com/): 

* The first one, `data_exploration_colab.ipynb`, takes the raw data (around 500k data points) and creates train, validation and test datasets, with X days of historic depth,and Y hours for prediction.
* The second one, `seq2seq_prediction_attention_colab.ipynb`, implements an state-of-the-art seq2seq model with attention and teacher-forcing. 

The code in both notebooks is ready to be implemented as a library, but I've rather preferred to leaeve everything in the notebook (for now), so as to ease its readability. This may change sooner than later =)

# Instalation and running instructions

The notebooks are prepared to be run directly on Google Colab, without any further requirement. I make use of Tensorflow 2.x, as well as standard libraries such as Pandas, Numpy or Matplotlib. So it should be easy to run locally or in another platform.

In order to reproduce the results, follow these steps:

1. First obtain the raw data sending an email to <diabetes.artificialintelligence@gmail.com>. Extact the contents of the zip file into your Google Drive; I recommend using the path: _**path_to_data=My Drive/Colab Notebooks/sugar_level_prediction/data/**_.

2. Upload the notebooks to your Google Drive, (for instance, to the path _**My Drive/Colab Notebooks/sugar_level_prediction/expore/**_, but this is not a requirement), and open them with Google Colab.  

3. Run the `data_exploration_colab.ipynb`. You can choose the number of days to use for the features (history), as well as the number of hours for the target (future prediction). Running this notebook will create a folder _**path_to_data/processed/**_, with `raw_data.csv` file containing the data after small preprocessing (around 500k data points). It will also add \*.npy files with the train, validation and test datsets (see section [Datasets](#datasets) for more details). Please note that this process can take more than an hour, dependign on the number of days you select for the history. Typically, 350k sequences will be created for training, 40k for validating and another 40k for testing. Make sure you have enough space in your Google Drive (for 6 days of history, the train dataset takes 4 GB).

4. Run the `seq2seq_prediction_attention_colab.ipynb`. The model is implemented in OOP style, using the [Keras](https://www.tensorflow.org/api_docs/python/tf/keras) API of Tensorflow, mostly. A base set of values for the parameters is provided in the Taining section, as well as several cells to run seq2seq vanilla models, teacher-forcing and various mechanisms of attention. See more details in the [Modeling](#modelling) section.

## Dataset

The train, validation and test datasets consit of sequences of `history`+`future` points, with 5 features (i.e. tensors of shape=(num_sequences, `history`+`future`, 5)) : 

* time interval: days counted starting from the end of the `history` of the sequence. Thus, for points in the `history`, this feature takes negative values, while for points in the `future`, it's positive. 
* hour: hour of the day
* day of week: day of the week in numbers ('Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6)
* patient_id: numeric identifier for the patient, starting at zero.
* sugar level: recorded sugar level, scaled with min/max scaler.

Additionally, we add two columns for the scale (min and max) of every sequence, so that the actual shape of the datasets is (num_sequences, `history`+`future`+2, 5). The scale is thus repeated 5 times (one for each feature).

The raw data is freely available upon email request to <diabetes.artificialintelligence@gmail.com>

# Modeling

The architecture consists of:

* `ProcessInput` module: It creates embeddings for categorical variables, concatenates them with numerical ones, and pass the result through a Dense layer (this last layer being optional). Here, we could also add a block of 1-dimensional Convolutional Neural Networks, to feed subsequent modules with an enriched set of features.

* `Encoder` module: an stack of Long-short term memory (LSTM) recurrent networks, that retrieves output sequences (of length equal to the input sequences), as well as the last states.

* `Attention` module: we implement the three most used scoring functions for attention, that is, additive, general and scaled dot-product. Concerning the use of the attention vector itself, we follow [Luong et al.](https://arxiv.org/abs/1508.04025), in which the context vector is calculated for the current decoder step, and concatenated with decoder state to make the predicition. We also feed the context vector as input to the following decoding step.

* `Decoder` module: an stack of LSTM, wich may optionally use attention, as well as teacher forcing. When using attention, the output of the decoder is concatenated with the attention context vector, and pass thorugh a MLP.

  Teacher focing is implemented with a Bernoulli random variable, that allows one to choose between prediction or ground-truth at step (t-1). If the Bernoulli probability equals 1, prediction at step (t-1) is fed to the decoder; on the contrary, if the probability is 0, it uses the ground truth. Intermediate probabilities allow one sampling between these two vectors. It would be interesting to implement a scheduling sampling procedure (Curriculum Learning). 

* `Seq2seq` module: it brings all modules together, and provides custom functions to fir, predict and evalute. It also has a method to rescale normalized sequences, useful for calculating metrics and plotting. 

In the experiemnts (made with the first 10 patients of the dataset), no improvements are observed when using the attention mechanism, but rather an increment in the time required for training. However, I have not extensively covered the hyper-parameter space, and thus there is room for improvement. Also, these models might require a larger amount of data.

Regarding teacher forcing, since the task is to generate sequences without making use of the ground truth, I do observe a deterioration in the forecasting (except when using the prediciton at (t-1) to feed the decoder at step t, instead of the ground truth). 


# Metrics

I define Mean Absolute Error (MAE) and Mean Absolute Precentage Error (MAPE) metrics for the predicted sequence, that is, averaged across the whole predicted sequence:

${\rm MAE} := \frac{1}{L}\sum_{t=1}^{L} |y_t - \hat{y}_t|$ (mg/dl)

${\rm MAPE} := \frac{1}{L}\sum_{t=1}^{L} \left|\frac{y_t - \hat{y}_t}{y_t}\right|\cdot 100 $ (%)

where $L$ is the number of steps in the predicted sequence. 

I average these metrics across all sequences in the test dataset(~40k series), and provide also its standard deviation. Generally, metric distributions present a tail of bad preforming predictions, which shifts the mean away from the median. 

It would be interesting to add uncertainty to the prediction, and see if values with higher uncertainty are those performing worse.


# Results 

Predict BG 3 hours in advance:

  * Naive Baseline (last value in sequence):

    MAE=39.0 mg/dl, std(MAE)=29.3 mg/dl; MAPE=28.5 %, std(MAPE)=23.4 %
    
  * 1 day of historic data: 
  
    MAE=32.6 mg/dl, std(MAE)=21.8 mg/dl; MAPE=24.7 %, std(MAPE)=19.0 %

  * 3 day of historic data: 
    
    MAE=31.5 mg/dl, std(MAE)=20.3 mg/dl; MAPE=24.5 %, std(MAPE)=18.9 %
    
  * 6 day of historic data: 
  
    MAE=30.9 mg/dl, std(MAE)=20.4 mg/dl; MAPE=23.0 %, std(MAPE)=16.5 %

Predict BG 1 hour in advance:

  * Naive Baseline (last value in sequence):

    MAE=22.5 mg/dl, std(MAE)=19.9 mg/dl; MAPE=16.4 %, std(MAPE)=15.8 %

  * 1 day of historic data: 
  
    MAE=20.0 mg/dl, std(MAE)=15.7 mg/dl; MAPE=16.0 %, std(MAPE)=15.9 %
    
  * 3 day of historic data: 
    
    MAE=18.5 mg/dl, std(MAE)=15.1 mg/dl; MAPE=14.2 %, std(MAPE)=13.9 %
    
  * 6 day of historic data:

    MAE=18.3 mg/dl, std(MAE)=14.9 mg/dl; MAPE=13.9 %, std(MAPE)=13.3 %
    
# Further steps

There are a number of things that can be done:

* Add some baselines, such as ARIMA, SVMs, random forest, etc.

* Add dropout at the output of the encoder and decoder LSTM stacks.

* Allow for the use of GRU, as well as LSTMs. This will speed up the computations.

* Add skip connections to the encoder and decoder LSTM stacks.


* Create a module of one-dimensional Convolutional Neural Networks (CNN) to extract features from the input, that are then feed to the encoder. I've played with that, but not systematically. 

* It would also be interesting to use Dialeted CNN for the encoder, instead of LSTMs.

* Add curriculum learning with some scheduling procedure for teacher-forcing.

In general, any idea that you might have will be very welcomed!

# Contribute

If you have a device that measures your BG automatically (either flash or a CGM), please consider donating your data for this research project! Just send an email to <diabetes.artificialintelligence@gmail.com>, and I will answer you ASAP.

If you like coding and want to fix a bug or add some functionally, send a Pull Request to the develop branch with your work! 