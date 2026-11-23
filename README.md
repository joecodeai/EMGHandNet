# EMGHandNet

Hand-gesture recognition from surface EMG. Raw muscle signals from the forearm are cleaned,
windowed and classified into hand movements with a CNN + BiLSTM.

## Result

| Dataset | Channels | Classes | Test samples | Test accuracy |
|---|---|---|---|---|
| [Ninapro DB2](https://ninapro.hevs.ch/) | 12 sEMG | 50 movements | 3,919 | **87.1%** |

## Approach

1. **Preprocessing** (`DB2_Preprocessing.ipynb`, `Preprocess_db2.ipynb`): load the Ninapro `.mat` files,
   drop rest periods, filter and normalise each channel, and cut the signal into fixed-length windows
   labelled with the restimulus class.
2. **Tabular export** (`convert_to_table.ipynb`, `db2_table.ipynb`): flatten windows into train/test tables
   (`example.csv` shows the raw layout).
3. **Model** (`DB2_Final.ipynb`): each window is split into sub-sequences; a time-distributed 1D CNN
   (Conv1D, batch norm, pooling) extracts local features and a bidirectional LSTM models how they change
   over time, followed by dense layers and softmax. A GRU variant was also tried.
4. **DB1** notebooks (`DB1.ipynb`, `db1_preprocess.ipynb`, `db1_train.ipynb`) run the same pipeline on
   Ninapro DB1 (10 channels).

## Run

```bash
pip install tensorflow numpy pandas scipy scikit-learn matplotlib jupyter
jupyter lab
```

Download Ninapro DB2 from the official site (registration required) and point the preprocessing
notebook at the extracted `.mat` files. The data is not included in this repository.

## Stack
Python · TensorFlow / Keras · NumPy · pandas · SciPy · scikit-learn

## Licence
MIT. Ninapro data is subject to its own terms of use.
