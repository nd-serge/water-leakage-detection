
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

pio.renderers.default = "browser"


DATASET_PATH = "../../data/dataset/df_ID_"
MODELS =  "../../data/models/"
DATA_PATH =  "../../data/detection_fuite.csv" 
x = "valeur_active"
RANDOM_STATE = 42


ID = "5"
X = pd.read_parquet(DATASET_PATH+ID)
print("Data Loaded!")


print("X Shape :", X.shape)

features = ["valeur_active", "resid", "seasonal", "trend"]



def baseline_model(X):
    q_10 = X["resid"].quantile(0.1)
    q_90 = X["resid"].quantile(0.9)
    X["is_leakage"] = ~X["resid"].between(q_10, q_90)
    return X

X_pred_baseline = baseline_model(X)

X_pred_baseline["is_leakage"].value_counts()

scaler = StandardScaler()
np_scaled = scaler.fit_transform(X)
data = pd.DataFrame(np_scaled)



X_features = X[features]

# ## Isolation Forest
n_estimators = 100  # Number of trees
contamination = "auto"  # Expected proportion of anomalies


# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=n_estimators,
                            contamination=contamination,
                            random_state=RANDOM_STATE)
iso_forest.fit(X_features)



# Calculate anomaly scores and classify anomalies
data = X_features.copy()
data['anomaly'] = iso_forest.predict(X_features)
data['anomaly_score'] = iso_forest.decision_function(X_features)

data['anomaly'].value_counts()



def train_iforest(n_estimators=100, contamination=0.075, X=X_features, 
          random_state=RANDOM_STATE, boostrap=True, max_samples=512,
                 title="iForest"):

    iso_forest = IsolationForest(n_estimators=n_estimators,
                                contamination=contamination,
                                random_state=random_state,
                                max_samples=max_samples,
                                bootstrap=boostrap)
    iso_forest.fit(X)

    data = X.copy()
    data['anomaly'] = iso_forest.predict(X)
    data['anomaly_score'] = iso_forest.decision_function(X)

    print(data['anomaly'].value_counts())

    return data


new_data = train_iforest()


# ### DBSCAN


dbscan = DBSCAN(eps=20, min_samples=10)
dbscan.fit(X[features])


# Extract labels and core samples
labels = dbscan.labels_
data = X[features].copy()
outliers = X[labels == -1]
data["is_noise"] = labels


# ## Evaluate

X_leaks_baseline = X_pred_baseline[X_pred_baseline["is_leakage"] == True]["resid"]
X_leaks_iforest = new_data[new_data["anomaly"] == -1]["resid"]
X_leaks_dbscan = outliers["resid"]




fig = go.Figure()

# Raw data
fig.add_trace(go.Scatter(
    x=X.index,
    y=X["resid"],
    mode="markers",
    name="Raw data",
    marker=dict(
        color="black",
        size=30,
        opacity=0.8
    )
))

# Baseline
fig.add_trace(go.Scatter(
    x=X_leaks_baseline.index,
    y=X_leaks_baseline.values,
    mode="markers",
    name="Baseline",
    marker=dict(
        symbol="x",
        color="teal",
        size=15
    )
))

# Isolation Forest
fig.add_trace(go.Scatter(
    x=X_leaks_iforest.index,
    y=X_leaks_iforest.values,
    mode="markers",
    name="Isolation Forest",
    marker=dict(
        symbol="triangle-up",
        color="red",
        size=20
    )
))

# DBSCAN
fig.add_trace(go.Scatter(
    x=X_leaks_dbscan.index,
    y=X_leaks_dbscan.values,
    mode="markers",
    name="DBSCAN",
    marker=dict(
        symbol="cross",
        color="purple",
        size=7
    )
))

fig.update_layout(
    title="Residual-Based Detection of Water Leakages Over Time",
    xaxis_title="Date",
    yaxis_title="Residuals",
    template="plotly_white",
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()



