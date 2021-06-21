from utils.preprocess import Processor
from models.models import DNN_Model


if __name__ == "__main__":
    saved_model = 'saved_models/eurusd.h5'
    train_data_dir = 'data/2020.parquet'
    test_data_dir = 'data/2021.parquet'
    num_features = 7
    num_time_steps=30
    future_window_size=7

    processor = Processor(num_time_steps=num_time_steps, num_features=num_features, future_window_size=7)
    X_train, y_train = processor.get_data_training(filename=train_data_dir)
    X_test, y_test = processor.get_data_training(filename=test_data_dir)
    
    # predicct the result
    model = DNN_Model(model_path=saved_model)
    y_predictions = model.predict(X_test)
    print(X_test.shape)
    result = processor.invert_prediction_result(X_test, y_predictions)
    print(result)
