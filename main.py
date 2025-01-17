from credit_score_predictor import CreditScorePredictor

if __name__ == "__main__":
    predictor = CreditScorePredictor("clientes.csv")
    predictor.preprocess_data()
    predictor.analyze_data()
    predictor.train_models()
    predictor.evaluate_models()