import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd
from prometheus_client import make_asgi_app, Counter
app = FastAPI()

survived_counter = Counter("survived", "Number of passengers that survived")
not_survived_counter = Counter('not survived', 'Number of passengers that did not survive')
@app.post("/titanic")
def prediction_api(pclass: int, sex: int, age: int) -> bool:
    # Load model
    titanic_model = joblib.load("model_titanic.joblib")

    # predict
    x = [pclass, sex, age]
    prediction = titanic_model.predict(pd.DataFrame(x).transpose())

    if prediction[0] == 1:
        survived_counter.inc()
    else:
        not_survived_counter.inc()
    return prediction[0] == 1

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
