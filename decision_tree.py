from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Charger les données
data = load_breast_cancer()

X = data.data
y = data.target

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Céer le modèle 
model = DecisionTreeClassifier(random_state=42)

# Entrainer le modèle
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Evaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

# Validation Croisée
from sklearn.model_selection import cross_val_score

scores =  cross_val_score(model, X, y, cv = 5, scoring= "accuracy")
print("Scores de Validation croisée :", scores)
print("Accuracy moyenne :", scores.mean())

# Detection du surapprentissage
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Accuracy entrainement :", train_accuracy)
print("Accuracy test :", test_accuracy)
print("Ecart :", train_accuracy-test_accuracy)

# Optimisation des paramètres avec GridSearchCV
from sklearn.model_selection import GridSearchCV

## Définissons les paramètres à tester
param_grid = {
    "max_depth": [2, 3, 4, 5, 6,
8,  10, None],
    "min_samples_split": [2, 5,
10, 20],
    "min_samples_leaf": [1, 2, 5,
10]
}


## Créer GridSearchCV
arbre = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator= arbre,
    param_grid=param_grid,
    cv=5,
    scoring= "accuracy",
    n_jobs= 1
)

## Lancer la recherche
grid_search.fit(X_train, y_train)

print("Meilleurs paramètres :",  grid_search.best_params_)
print("Meilleur score :", grid_search.best_score_)

# Meilleur modèle
best_model = grid_search.best_estimator_

print("Score train :", best_model.score(X_train, y_train))
print("Score test :", best_model.score(X_test, y_test))
ecart = best_model.score(X_train, y_train)-best_model.score(X_test, y_test)
print("Ecart :", ecart)

# Mesurer le surapprentissage fold par fold avec les resultats de la validation croisée

from sklearn.model_selection import cross_validate
resultats = cross_validate(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    return_train_score=True

)

## Récupérer les scores d'entrainement, de validation pour les 5 folds.
for i in range(5):
    score_train = resultats["train_score"][i]
    score_validation = resultats["test_score"][i]
    ecart = score_train - score_validation
    print(f"Fold {i+1}")
    print(f"Score entrainement : {score_train:.4f}")
    print(f"Score validation : {score_validation:.4f}")
    print(f"Ecart : {ecart:.4f}")
    print("-" * 30)
    ecart = [0.0220, 0.0714, 0.0110, 0.0577, 0.0467]  
    ecart_moyen = sum(ecart)/len(ecart)
    print(f"Ecart_moyen : {ecart_moyen:.4f}")
    









