import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Label encoder: adds a way to give value to strings
# standarad scaler, added to allow for scaling of data
from sklearn.preprocessing import LabelEncoder, StandardScaler
# mpatches make data readable on visulization essnetially adding 
# legend and color coding on plot
import matplotlib.patches as mpatches
# Neural Network Library
from sklearn.neural_network import MLPClassifier
# accuracy, precision, recall, f1-score,
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your datasets
data = pd.read_csv('gym_members_exercise_tracking_test.csv')
orignial = pd.read_csv('gym_members_exercise_tracking.csv')
label_encoder = LabelEncoder()

# same as KNN project (only for visulization)
def get_colors(gender_data):
    return ['red' if gender == 0 else 'blue' if gender == 1 else 'black' for gender in gender_data]

def plot_pca(ax, x, y, title):
    # PCA to get 3d vectors of each point
    pca = PCA(n_components=3)
    pca.fit(x)
    PCAX = pca.transform(x)
    
    # determin color via (0 = red, 1 = boy, neither = Black)
    colors = get_colors(y)
    
    # scatter all points on to the figure
    scatter = ax.scatter(PCAX[:, 0], PCAX[:, 1], PCAX[:, 2], color=colors, s=10)
    
    # Add a legend
    Female = mpatches.Patch(color='red', label='Female')
    Male = mpatches.Patch(color='blue', label='Male')
    Undetermined = mpatches.Patch(color='black', label='Undetermined')
    ax.legend(loc='upper left', handles=[Female, Male, Undetermined])
    
    # set plot labels and title
    ax.set_title(title)
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.set_zlabel('Third component')


# SAME EXACT AS PROJECT 2 FOR PRE PROCESSING METHOD
# assign value to workout types For all given data types
data['Workout_Type'] = label_encoder.fit_transform(data['Workout_Type'])

orignial['Workout_Type'] = label_encoder.fit_transform(orignial['Workout_Type'])
orignial['Gender'] = label_encoder.fit_transform(orignial['Gender'])

# known data vs unknown data, need to seperate
known_data = data[data['Gender'].notna()].copy()
unknown_data = data[data['Gender'].isnull()].copy()

# covert gender (male/female) to 1 or 0
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Create figure for PCA plots
fig = plt.figure()



# Original data(no missing values of gender)
x_original = orignial.drop('Gender', axis=1)
y_original = orignial['Gender']
ax1 = fig.add_subplot(131, projection="3d")
plot_pca(ax1, x_original, y_original, 'Original Male vs Female in Gym')



# Data with missing values
x_data = data.drop('Gender', axis=1)
y_data = data['Gender']
ax2 = fig.add_subplot(132, projection="3d")
plot_pca(ax2, x_data, y_data, 'Missing Male vs Female in Gym')



# y_train contains either male or female, x contains everything else(except gender)
# X_unknown contains same as x from earlier, but no gender given
X_train = known_data.drop('Gender', axis=1)
y_train = known_data['Gender']
X_unknown = unknown_data.drop('Gender', axis=1)

y_train = label_encoder.fit_transform(y_train) # change from male/female to 0/1

# Normalize (scale) features for training
# This gave much better results when scaled, and its the same as KNN
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_unknown_scaled = scaler.transform(X_unknown)

# Train MLP on scaled data (no PCA for training unlike my knn model from project2)
# used defualt activation "relu" and solver "adam", and amax iter of 500 to esnure covergence
# the number of hidden layers was abritarary, no reason behind decision
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# predict gender for unknown data points
predictions = mlp.predict(X_unknown_scaled)

# put original values into a numpy array 
original_comparson = np.array(orignial['Gender'].iloc[760:])


# get axis of last plot
PredicitedGraph = fig.add_subplot(133, projection="3d")

# Apply PCA for visualization on the known data
pca_known_final = PCA(n_components=3)
PCAX_known_final = pca_known_final.fit_transform(X_train_scaled)

# Apply PCA for visualization on the unknown data
pca_unknown_final = PCA(n_components=3)
PCAX_unknown_final = pca_unknown_final.fit_transform(X_unknown_scaled)

colors_known = get_colors(y_train)  # get their colors
colors_unknown = get_colors(predictions)

# scatter across plot, o for original/true, triangle for predicted
scatter_known = PredicitedGraph.scatter(PCAX_known_final[:, 0], PCAX_known_final[:, 1], 
                                        PCAX_known_final[:, 2], color=colors_known, s=10, marker='o')
scatter_unknown = PredicitedGraph.scatter(PCAX_unknown_final[:, 0], PCAX_unknown_final[:, 1], 
                                          PCAX_unknown_final[:, 2], color=colors_unknown, s=10, marker='^')

# Add a legend to the final graph
Female = mpatches.Patch(color='red', label='Female')
Male = mpatches.Patch(color='blue', label='Male')
PredicitedGraph.legend(loc='upper left', handles=[Female, Male])
    
# set plot labels and title for the final graph
PredicitedGraph.set_title("predicted values for male vs female in gym")
PredicitedGraph.set_xlabel('First component')
PredicitedGraph.set_ylabel('Second component')
PredicitedGraph.set_zlabel('Third component')


# Show plot(shows all 3)
plt.show()


# classifcation model, shows accuracy for general right predicitons
# precision predicts the amount of times it got male right
# recall shows all the positive instances
# F1 score shows the harmonic mean of precision and recall
accur = accuracy_score(original_comparson, predictions)
precict = precision_score(original_comparson, predictions)
recal = recall_score(original_comparson, predictions)
f1sc = f1_score(original_comparson, predictions)

print("Accuracy = ", accur * 100, "%", "\nPrecision = ", precict)
print("Recall =", recal, "\nF1-Score = ", f1sc)
