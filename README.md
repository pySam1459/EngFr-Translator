# Translator

A PyTorch implementation of an English to French translator by Samuel Barnett.</br>
The implementation loosely follows from the 'Attention Is All You Need' paper, which initially introducing the transformer architecture [Link](https://arxiv.org/pdf/1706.03762.pdf).

## Dataset
This project uses Kaggle.com's English to French translation dataset [Link](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset).</br>
The dataset contains 22,520,319 english-french sentence pairs, approximately 8.2GB, but it should be noted that some of these sentence pairs are of poor quality, with respect to their translation. And so, I have employed some simple cleaning techniques to increase the quality of the final dataset (assumed sentences should be of roughly equal length)</br>
Dataset path is specified in `.env` file: 
```yaml
DATASET_PATH = "PATH/TO/DATASET"
```

### Preprocessing
Before training of the translator, the dataset needs to be tokenized, partitioned and cleaned.

#### Tokenization
The `tiktoken` library is a useful tokenization library, famously used by OpenAI for their GPT models. My translator model uses a slightly modified version of the tokenizer used by GPT-2, with additional start and padding special tokens.</br>
The script `./preprocess/tokenizer.py` tokenizes the sentences contained in the raw dataset, storing the results in `DATASET/PATH/train.h5`.</br>
The [HDF5](https://docs.h5py.org/en/stable/) file format is great for storing large datasets and grants easy access to the data, providing an interface similar to slicing a NumPy array.

#### Partitioning
I partitioned the tokenized dataset for a couple of reasons:
 - The dataset is large enough to take up most of system memory,
 - To create batches, sentences should be of roughly equal length, to reduce the number of padding tokens.
The dataset is partitioned into separate .h5 files with all english sentences in the 


#### Cleaning
As mentioned above, the original dataset contains many sentence pairs which have no translation equivalence, and I assume that these cases are caused by errors in the program which originally created the dataset.</br>
To remove these cases from the dataset by hand would be an intractable task, as there are 22.5M sentence pairs, however a simple length comparison can be used to discard obvious error cases. </br>
If the french translation's length falls outside of some predetermined interval (based on the length of the english sentence), then that sentence pair is assumed to be incorrect and is discarded.

## Model
The translator model follows a similar architecture to the transformer outlined in the original paper mentioned above. The transformer tutorial by Andrej Karpathy [Link](https://youtu.be/kCc8FmEb1nY), was a great introduction into transformer programming and heavily inspired this project.

### Parameters
With the hyperparameters outlined below, the total parameter count is approximately = 84.4M</br>
</br>
| Hyperparameter |  |
| --- | --- |
| d_model | 384 |
| n_layers | 6 |
| n_head | 6 |
| d_head | 64 |
| dropout | 0.1 |
| lr | 1e-4 |
| batch size | 64 |