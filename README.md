# Présentation d'Imagy

## A quoi sert Imagy ?
Imagy est une API permettant l'augmentation de jeux de données comportant des images. L'augmentation de jeux de données (ou data augmentation) est une technique souvent utilisée en Machine Learning et Deep learning afin de pouvoir entrainer les modèles sur davantages d'échantillons.

## Comment fonctionne Imagy ?
Afin d'augmenter la taille du jeu de données, Imagy applique des transformations à chacune des images soumises par l'utilisateur afin de créer de nouvelles images. Parmi les transformations supportées :
- Flip : l'image est transformée en deux images, respectivement inversées verticalement et horizontalement
- Colors : l'image RGB est transformée en trois images, respectivement en nuances de rouge, vert, bleu et gris
- Brightness : deux nouvelles images sont générées avec des niveaux de luminosité aléatoires
- GaussianNoise : une image bruitée est générée
- Rotate : deux nouvelles images sont générées en appliquant des rotations d'angles aléatoires
- Shift : deux nouvelles images sont générées en appliquant des translations aléatoires


# Utiliser Imagy


## Lancer l'application
L'application peut être lancée en utilisant Docker. Après avoir pull le repository, ouvert un terminal et être allé sur le dossier correspondant au repository, taper
```Python
docker build -t imagy .
docker run -p 5000:5000 imagy
```
L'application devrait alors être disponible à l'adresse http://localhost:5000

## Utiliser l'application
Sur la page d'accueil de l'application, des cases à cocher permettent de sélectionner les transformations souhaitées. Certaines transformations demandent davantage de temps de calculs et les sélectionner toutes demande environ 3 minutes par image.
Le bouton "choisir un fichier" permet de charger le jeu de données à augmenter. Il est possible de charger une image seule ou un fichier zip contenant plusieurs images.
Enfin, le bouton submit lance le code en backend qui se charge d'augmenter le jeu de données. Selon le nombre d'images soumises, cela peut demander du temps, soyez patient !

Une fois les résultats prêts, vous serez redirigé vers une nouvelle page. Il vous suffira de cliquer sur le bouton download pour récupérer un zip avec les nouvelles images générées. Notez que les images originales ne sont pas présent dans ce nouveau dossier : celui-ci comporte uniquement les nouvelles images générées et a vocation a être fusionné avec le jeu de données initial.

# Code d'Imagy

## Contribuer au projet
Toute contribution est la bienvenue. Veuillez coder vos modifications dans une nouvelle branche feature/ et soumettre une merge request dans la branche dev.

## Se placer dans un environnement virtuel, tester l'application
Après avoir pull le repository, ouvrez un terminal et tapez (sous linux/wsl):
```Python
python -m venv env
source ./env/bin/activate
python -m pip install -r requirements.txt
```

Pour tester l'impact de vos changements sur l'application, vous pouvez taper la commande:
```Python
flask run
```

## Architecture et fonctionnement du code
Le script app.py utilise Flask pour construire les routes de l'application. Il s'appuie sur les fichiers html et css présents respectivement dans templates/ et static/
Lorsqu'un dataset est soumis par l'utilisateur, celui-ci est téléchargé dans static/images_input/. app.py le décompresse puis appelle la fonction generate_all de main.py, qui est le point d'entrée du backend.

generate_all applique chacune des transformations requises par l'utilisateur à chacune des images du dataset décompressé. Les résultats sont sauvegardés dans static/images_output/, auquel app.py accédera pour compresser ce dossier et le renvoyer au frontend.

Chacune des transformation correspond à une classe présente sous image_editor/transformations/. Elles héritent d'une classe parente Image, présente sous image_editor/preparations/, en charge de créer un objet Image contenant des méthodes générales (charger, afficher, sauvegarder...). Elles possèdent également toutes une méthode generate() qui crée une liste d'objets Image transformées.









