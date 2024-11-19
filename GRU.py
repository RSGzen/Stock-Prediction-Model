import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class StockPredictionModel:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model = None

    def load_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df.astype(float)

        x = df.drop(["Adj Close", "Close"], axis=1)
        y = df["Adj Close"].values.reshape(-1, 1)

        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))

        self.x_transformed = self.x_scaler.fit_transform(x)
        self.y_transformed = self.y_scaler.fit_transform(y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_transformed, self.y_transformed, test_size=0.2, random_state=42)

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))

    def build_model(self):
        self.model = Sequential([
            GRU(units=100, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2])),
            Dropout(rate=0.3),

            GRU(units=100, return_sequences=True),
            Dropout(rate=0.3),

            GRU(units=50, return_sequences=True),
            Dropout(rate=0.3),

            GRU(units=50),
            Dropout(rate=0.3),
            Dense(units=1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def train_model(self, epochs=30, batch_size=24):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

        self.y_test_original = self.y_scaler.inverse_transform(self.y_test)
        self.y_pred_original = self.y_scaler.inverse_transform(self.y_pred)

        return self.y_test_original, self.y_pred_original

    def back_test(self, test_data, pred_data):
        mse = mean_squared_error(test_data, pred_data)
        mae = mean_absolute_error(test_data, pred_data)
        r2_score_val = r2_score(test_data, pred_data)
        variance = explained_variance_score(test_data, pred_data)

        print("Mean Square Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R2-Score:", r2_score_val)
        print("Explained Variance Score:", variance)

        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(test_data, label='Actual')
        plt.plot(pred_data, label='Predicted')
        plt.title('Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(2, 1, 2)
        residuals = test_data - pred_data
        plt.plot(residuals, label='Residuals')
        plt.title('Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residual')
        plt.legend()
        plt.tight_layout()
        plt.show()


# Usage Example
if __name__ == "__main__":
    model = StockPredictionModel(ticker="AAPL", start_date="2010-01-01", end_date="2020-01-01")
    model.load_data()
    model.build_model()
    model.train_model(epochs=30, batch_size=24)
    y_test_original, y_pred_original = model.predict()
    model.back_test(y_test_original, y_pred_original)

#https://medium.com/@udaytripurani04/stock-market-predictions-using-lstm-and-gru-models-with-python-ca103183dbc0
