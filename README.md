# System sztucznej inteligencji - Projekt Zaliczeniowy

## Źródło danych

W projekcie zostały wykorzystane dane dotyczące ekonomii gry Path of Exile. 
Strona [poe.ninja](https://poe.ninja) zajmuje się agregacją tych danych w czasie rzeczywistym. 
Wybrane zostały dane dotyczące wymiany walut z 2 ostatnich lat - 5 cykli gry nazywanych ligami - ang. leagues.

Dane zostały pobrane z zakładki Data dumps na stronie [poe.ninja/poe1/data](https://poe.ninja/poe1/data).     
Dane podlegają licencji cc-by-sa.

Przykład danych z pliku Keepers.currency.csv:

```csv
League;Date;Get;Pay;Value;Confidence
Keepers;2025-10-31;Chromatic Orb;Chaos Orb;0.2;High
Keepers;2025-11-01;Chromatic Orb;Chaos Orb;0.19231;High
Keepers;2025-11-02;Chromatic Orb;Chaos Orb;0.19231;High
Keepers;2025-11-03;Chromatic Orb;Chaos Orb;0.16667;High
Keepers;2025-11-04;Chromatic Orb;Chaos Orb;0.1497;High
```

## Przygotowanie danych

Dane uległy transformacji - uzależniono wszystkie waluty od "Chaos Orb" - głównej waluty do wymiany w grze. Zawiera ona najwięcej danych o najlepszym współczynniku Confidence. 
Data zamieniana jest na liczbę dnia ligii. Każda liga jest zapisywana w osobnym DataFrame. 

Przekonwertowane dane mają charakter szeregów czasowych. Dane do treningu zostaną poukładane w odpowiednie serie dni. Celem modeli będzie przewidzenie wartości na podstawie poprzednich dni, np. wartość waluty dnia 15, mają informacje o poprzednich 14 dniach. 

```python
def read_currency_data(filepath: str, start_date: str):
    """
    reads file and cuts the unnecessary data: League and Confidence.
    Date is changed into day of the league, to transform date into number.
    returns DataFrame
    """
    df = pd.read_csv(filepath, sep=';', skipinitialspace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Pay'] == 'Chaos Orb']
    df['Day of league'] = (df['Date'] - pd.to_datetime(start_date)).dt.days + 1
    columns_to_drop = ['League', 'Date', 'Pay', 'Confidence']
    df = df.drop(columns=columns_to_drop)
    df = df.reset_index(drop=True)
    return df
```

Dane przycinane są do 100 dni, aby wyrównać zakresy. Jedna liga okazała się zbyt długa w porównaniu do reszty lig. 

```python
def transform_currency_data(df):
    """
    Drop with all NaN
    Drop with 5 NaN's in Row - bad data
    Fill NaN's with data from day before, and fill first day's
    Cuts data to 100 days (Settlers is too large)
    returns DataFrame
    """
    df = df.pivot(index='Day of league', columns='Get', values='Value')
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=1, thresh=len(df) - 4)
    df = df.ffill().bfill()
    df = df[df.index <= 100]
    return df.T
```

Wybrana została waluta "Divine Orb", natomiast okno czasowe to 14 dni.
```python
ITEM = "Divine Orb"
WINDOW = 14
```

Próbka danych wykorzystywanych do trenowania, są to listy o długości okna. 
```
Dane treningowe: (344, 14)
[[1.         2.31578947 2.84210526 ... 4.57355263 4.39473684 4.07894737]
 [2.31578947 2.84210526 3.60526316 ... 4.39473684 4.07894737 4.02631579]
 [2.84210526 3.60526316 3.76315789 ... 4.07894737 4.02631579 3.94736842]
 ...
 [2.69439421 2.67631103 2.62206148 ... 2.53164557 2.53164557 2.53164557]
 [2.67631103 2.62206148 2.60759494 ... 2.53164557 2.53164557 2.48101266]
 [2.62206148 2.60759494 2.60262206 ... 2.53164557 2.48101266 2.44122966]]
```

## Wykrywanie anomalii

Wykorzystano 4 modele do detekcji anomalii: IsolationForest, OneClassSVM, LocalOutlierFactor oraz EllipticEnvelope. 
Założono 0.1 zanieszczenia danych. 

![Wykres ilości anomalii w danych](images/anomaly_detection.png)

Najbardziej czułym modelem na usunięcie anomalii była sieć neuronowa. Reszta modeli nie zmieniała wyników, przy wyczyszczeniu danych. 

### Isolation Forest

![Isolation Forest](images/isolation_forest.png)

### One Class SVM

![OneClassSVM](images/oneclasssvm.png)

### 

![lof](images/lof.png)

### Elliptic Envelope

![elliptic envelope](images/elliptic_envelope.png)

### Oryginalne dane

![original data](images/original.png)

## Uczenie modeli 

Do znalezienia najlepszych parametrów, wykorzystano `GridSearchCV`, ze strategią `TimeSeriesCrossValidation`. Kryterium wyboru jest średni błąd `neg_mean_absolute_error`. 
Wyszukiwanie parametrów zostało przeprowadzone dla danych orignalnych oraz danych po anomalii. 

## RandomForest

### Parametry kalibracji 

```python
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
```

### Ocena

Wynik uczenia Random Forest dla danych oryginalnych: 

```
============================================================
Przetwarzanie danych: original_data
============================================================
Fitting 5 folds for each of 24 candidates, totalling 120 fits
  MAE:  4.01
  MAPE: 2.52%
  MSE:  31.31
  R²:   0.9131
  best_params: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
```

Uzyskany model osiągnął dobre wyniki predykcyjne. Współczynnik R2 na poziomie 0.9131 wskazuje na to, że model wyjaśnia 91% zmienności danych. 

![Random Forest](images/random_forest.png)

## Support Vector Regressor

### Parametry kalibracji 

```python
svr_param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [1, 10, 100],
    'epsilon': [0.05, 0.1],
    'gamma': ['scale', 0.1, 1]
}

svr_base = SVR()
svr_grid = GridSearchCV(
    estimator=svr_base,
    param_grid=svr_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
```

### Ocena

```
============================================================
Przetwarzanie danych: original_data
============================================================
Fitting 5 folds for each of 36 candidates, totalling 180 fits
  MAE:  2.72
  MAPE: 1.79%
  MSE:  16.60
  R²:   0.9540
  best params: {'C': 10, 'epsilon': 0.05, 'gamma': 'scale', 'kernel': 'linear'}
```

Model SVR jest najlepszym zastosowanym modelem w tym eksperymencie. Współczynnik R2 wyjaśnia 95% zmienności danych. Mean Square Error (MSE) jest prawie dwukrotnie mniejszy niż w przypadku Random Forest.  

![svr](images/svr.png)

## K Nearest Regressor

### Parametry kalibracji 

```python
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

knn_base = KNeighborsRegressor()
knn_grid = GridSearchCV(
    estimator=knn_base,
    param_grid=knn_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
```

### Ocena

```
============================================================
Przetwarzanie danych: original_data
============================================================
Fitting 5 folds for each of 72 candidates, totalling 360 fits
  MAE:  7.44
  MAPE: 4.85%
  MSE:  89.41
  R²:   0.7519
  best params: {'algorithm': 'auto', 'n_neighbors': 7, 'p': 2, 'weights': 'uniform'}
```

![kNN](images/knn.png)

## Sieć Neuronowa

### Parametry kalibracji 

```python
nn_configs = [
    {
        'layers': [64, 32, 16], 
        'dropout': 0.3,
        'lr': 0.001, 
        'activations': 'relu'
    },
    {
        'layers': [128, 64, 32, 16], 
        'dropout': 0.2, 
        'lr': 0.001,
        'activations': ['elu', 'elu', 'elu', 'relu']
    },
    {
        'layers': [8, 16, 32, 64, 32, 16, 8], 
        'dropout': 0.2,
        'lr': 0.0005, 
        'activations': 'relu'
    },
    {
        'layers': [32, 32, 32, 32, 32], 
        'dropout': 0.4,
        'lr': 0.001, 
        'activations': 'leaky_relu'
    },
]
```

### Ocena

```
============================================================
Przetwarzanie danych: original_data
============================================================
Testowanie: {'layers': [64, 32, 16], 'dropout': 0.3, 'lr': 0.001, 'activations': 'relu'}
  Val MAE: 0.1007
Testowanie: {'layers': [128, 64, 32, 16], 'dropout': 0.2, 'lr': 0.001, 'activations': ['elu', 'elu', 'elu', 'relu']}
  Val MAE: 0.0903
Testowanie: {'layers': [8, 16, 32, 64, 32, 16, 8], 'dropout': 0.2, 'lr': 0.0005, 'activations': 'relu'}
  Val MAE: 1.0480
Testowanie: {'layers': [32, 32, 32, 32, 32], 'dropout': 0.4, 'lr': 0.001, 'activations': 'leaky_relu'}
  Val MAE: 0.1651
  MAE:  3.98
  MAPE: 2.63%
  MSE:  26.92
  R²:   0.9253
  best params: {'layers': [128, 64, 32, 16], 'dropout': 0.2, 'lr': 0.001, 'activations': ['elu', 'elu', 'elu', 'relu']}
```

![Neural network](images/neural_network.png)

## System Ekspercki

OPISAĆ ZASADY UTWORZONE I METODY

### Logika rozmyta 

![Symulacja systemu eksperckiego](images/system_ekspercki.png)

## Wnioski

Najlepszym modelem okazał się Support Vector Regressor. 
Najgorszym modelem okazała się sieć neuronowa. 

Model LOF najlepiej poprawił działanie modelu. 

```
Eksperyment            Model     MAE    MAPE      MSE     R²
 Random Forest  IsolationForest  4.2183  2.6492  33.4782 0.9071
 Random Forest      OneClassSVM  4.0525  2.5586  30.8388 0.9144
 Random Forest              LOF  4.6732  2.9058  42.5053 0.8821
 Random Forest EllipticEnvelope  4.3143  2.7161  33.9479 0.9058
 Random Forest    original_data  4.0066  2.5183  31.3055 0.9131
           SVR  IsolationForest  2.6765  1.7679  16.1445 0.9552
           SVR      OneClassSVM  2.7969  1.8387  17.4588 0.9516
           SVR              LOF  2.5577  1.6769  15.2505 0.9577
           SVR EllipticEnvelope  2.8302  1.8597  17.5082 0.9514
           SVR    original_data  2.7183  1.7883  16.5969 0.9540
           KNN  IsolationForest  7.5045  4.8288  87.8985 0.7561
           KNN      OneClassSVM  7.3505  4.7869  88.3379 0.7549
           KNN              LOF  7.9715  5.1512  96.8545 0.7313
           KNN EllipticEnvelope  7.8936  5.0669  94.2837 0.7384
           KNN    original_data  7.4375  4.8459  89.4110 0.7519
Sieć neuronowa  IsolationForest 16.4209 10.8050 355.4596 0.0137
Sieć neuronowa      OneClassSVM  9.4845  6.3217 138.2944 0.6163
Sieć neuronowa              LOF  9.1974  6.2302 127.5432 0.6461
Sieć neuronowa EllipticEnvelope  9.1799  5.8987 139.4820 0.6130
Sieć neuronowa    original_data  3.9811  2.6279  26.9234 0.9253

================================================================================
NAJLEPSZE WYNIKI
================================================================================

Najlepszy wynik dla MAE:
  Eksperyment: SVR
  Model detekcji anomalii: LOF
  Wartość MAE: 2.5577

Najlepszy wynik dla MAPE:
  Eksperyment: SVR
  Model detekcji anomalii: LOF
  Wartość MAPE: 1.6769

Najlepszy wynik dla MSE:
  Eksperyment: SVR
  Model detekcji anomalii: LOF
  Wartość MSE: 15.2505

Najlepszy wynik dla R²:
  Eksperyment: SVR
  Model detekcji anomalii: LOF
  Wartość R²: 0.9577
```

## Autor
Kacper Wiącek, 99733, Informatyka II stopnia, grupa 4.8