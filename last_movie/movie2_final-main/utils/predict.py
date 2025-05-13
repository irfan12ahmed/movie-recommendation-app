import numpy as np

def predict_popularity(title, budget, runtime):
    return (budget / 1000000) + (runtime / 2)

def predict_revenue(budget, popularity):
    return budget * (popularity / 100 + 1)