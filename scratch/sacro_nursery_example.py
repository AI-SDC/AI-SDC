'''Train the nursery model in a disclosive and non-disclosive way'''
# %% Imports required
import sys
import os
import logging
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add AI-SDC to path - development copy. Can ignore if importing
# TODO: JIM, the path append can be removed assuming TRE users will import aisdc
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aisdc.attacks.dataset import Data
from aisdc.attacks.worst_case_attack import WorstCaseAttack, WorstCaseAttackArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

# Suppress messages from AI-SDC -- comment out these lines to see all the aisdc logging statements
logging.getLogger('attack-reps').setLevel(logging.WARNING)
logging.getLogger('prep-attack-data').setLevel(logging.WARNING)
logging.getLogger('attack-from-preds').setLevel(logging.WARNING)

# %% Define some constants
TEST_PROP = 0.7 # proportion of data used for testing the model. This would be set by the researcher

# These hyperparameters lead to a dangerously disclosive trained model
DISCLOSIVE_HYPS = {
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_depth': None,
    'bootstrap': False
}

# These make a much safer model
SAFE_HYPS = {
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_depth': 5,
    'bootstrap': True
}

# %% Get the dataset and preprocess -- this would be done by the researcher
data = fetch_openml(data_id=26, as_frame=True)
target_encoder = LabelEncoder()
target_vals = target_encoder.fit_transform(data["target"].values)
target_dataframe = pd.DataFrame({"target": target_vals})
feature_encoder = OneHotEncoder()
x_encoded = feature_encoder.fit_transform(data["data"]).toarray()
feature_dataframe = pd.DataFrame(
    x_encoded, columns=feature_encoder.get_feature_names_out()
)

# %% Make a train-test split of the data -- done by the researcher
trainX, testX, trainy, testy = train_test_split(
    feature_dataframe.values,
    target_dataframe.values.flatten(),
    test_size=TEST_PROP
)

#%% Training a disclosive model - done by the researcher
target_model = RandomForestClassifier(**DISCLOSIVE_HYPS)
target_model.fit(trainX, trainy)

# %% Evaluate target model performance -- don't actually need this for testing the model w.r.t
# disclosure, but something a researcher would almost certainly do
train_acc = accuracy_score(trainy, target_model.predict(trainX))
test_acc = accuracy_score(testy, target_model.predict(testX))
print(f'Train ACC: {train_acc:.2f}, test ACC: {test_acc:.2f}')


# %% Run worst case attacks on the disclosive model -- this would be done by the TRE
sdc_data = Data()
# Wrap the training and test data into the Data object
sdc_data.add_processed_data(trainX, trainy, testX, testy)
# Create attack args.
args = WorstCaseAttackArgs(n_dummy_reps=0, report_name=None)
# Run the attack
wca = WorstCaseAttack(args)
wca.attack(sdc_data, target_model)

# TODO: Yola please add calls to output stuff here!

# %%
#%% Training a safe model

target_model = RandomForestClassifier(**SAFE_HYPS)
target_model.fit(trainX, trainy)

# %% Evaluate target model performance
train_acc = accuracy_score(trainy, target_model.predict(trainX))
test_acc = accuracy_score(testy, target_model.predict(testX))
print(f'Train ACC: {train_acc:.2f}, test ACC: {test_acc:.2f}')


# %% Run worst case attacks on the disclosive model
sdc_data = Data()
sdc_data.add_processed_data(trainX, trainy, testX, testy)
# Create attack args. Note that n_dummy_reps=0 may cause an error. If so, either
# change it to 1, or update your branch (once we've merged the fix)
args = WorstCaseAttackArgs(n_dummy_reps=0, report_name=None)
wca = WorstCaseAttack(args)

# Suppress messages from AI-SDC
logging.getLogger('attack-reps').setLevel(logging.WARNING)
logging.getLogger('prep-attack-data').setLevel(logging.WARNING)
logging.getLogger('attack-from-preds').setLevel(logging.WARNING)

wca.attack(sdc_data, target_model)

# TODO: Yola please add calls to output stuff here!

# %%
