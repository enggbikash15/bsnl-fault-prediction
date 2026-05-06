import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate realistic telecom dataset

np.random.seed(42)

records = 50000

data = pd.DataFrame({

    'signal_strength': np.random.randint(10, 100, records),

    'fiber_loss': np.random.uniform(0.1, 6.0, records),

    'temperature': np.random.randint(20, 55, records),

    'traffic_load': np.random.randint(10, 100, records),

    'power_status': np.random.randint(0, 2, records),

    'equipment_health': np.random.randint(40, 100, records),

    'humidity': np.random.randint(30, 95, records)

})

# Intelligent fault logic

data['fault'] = (

    (data['signal_strength'] < 35) |

    (data['fiber_loss'] > 4) |

    (data['temperature'] > 45) |

    (data['traffic_load'] > 95) |

    (data['power_status'] == 0) |

    (data['equipment_health'] < 50)

).astype(int)

# Save dataset

data.to_csv("data/dataset.csv", index=False)

# Features

X = data[[
    'signal_strength',
    'fiber_loss',
    'temperature',
    'traffic_load',
    'power_status',
    'equipment_health',
    'humidity'
]]

# Target

y = data['fault']

# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# Save model

joblib.dump(model, "model.pkl")

print("✅ Telecom ML Model Trained Successfully")
