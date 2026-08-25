from sklearn.neighbors import KNeighborsClassifier

# Training dataset
# Features: [Fever, Cough, Fatigue, Difficulty_Breathing]
X = [
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 1]
]

# Labels
# 0 = No Condition
# 1 = Condition
y = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]

# Get input from user
print("Enter patient symptoms:")
fever = int(input("Fever (0 = No, 1 = Yes): "))
cough = int(input("Cough (0 = No, 1 = Yes): "))
fatigue = int(input("Fatigue (0 = No, 1 = Yes): "))
breathing = int(input("Difficulty in Breathing (0 = No, 1 = Yes): "))

k = int(input("Enter the value of K: "))

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X, y)

# New patient's data
new_patient = [[fever, cough, fatigue, breathing]]

# Prediction
prediction = knn.predict(new_patient)

# Display result
if prediction[0] == 1:
    print("\nPrediction: Patient HAS the medical condition.")
else:
    print("\nPrediction: Patient DOES NOT HAVE the medical condition.")