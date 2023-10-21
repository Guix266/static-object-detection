# Détection d'un objet statique 

- But : 
Détection d'un objet qui va entrer dans le champ de la caméra et rester statique pendant un certain temps.
Par exemple un sac, une poubelle.... Le programme entourre ces zones en rouge.

- Principe: Détection des objets statiques puis classification

## I. Soustraction de background :

Afin de détecter les objets qui restent statique pendant plusieurs secondes, on pratique ce que l'on appelle une "soustraction de fond".
Voici les étapes de l'algorithme:
1) En entrée, se trouvent les images correspondant aux frames de la vidéo. On traite les image une par une.
2) L'image est binarizée 2 fois selon la méthode de Zivkovich. Différents learning rates sont utiliséespour qu'un des background 
soient mis à jours plus lentement que l'autre ( un long, un court). On obtient alors deux images contenant les objets en mouvement.
3) On effectue un test sur chaque pixel des 2 background. Si celui-ci appartent à un objet en mouvement sur l'image avec les long 
learning rate mais pas à celle avec le court learning rate alors il appartion à un objet qui est arrivé sur l'image et reste imobile pendant 
un certain temps.
4) Si un groupe de plus de N pixels sont côte à côte alors on le concidère comme un objet. On l'entourre alors en rouge.

## II. Classification :
Une fois les objets statiques détectés nous les classifions comme baggages ou non. Pour cela nous avons entrainé un Convolutional Neural Netwark sur des images de sac, valises, train, personnes, .... Pour augmenter la robustesse de notre modèle nous avons utlisé un réseau de neurones préentrainé pour lequel nous n'avous entrainé que les dernières couches.

- Test du programme:
1) Choix du Dataset dans le programme main.py
2) On peut modifier les paramètres dans const.py : Les deux learning rates et les deux valeurs de Threshold
2) Lancer main.py en choisisant le dataset voulu pour travailler sur une vidéo, ou lancer main_url.py en indiquant le lien http voulu pour travailler sur un flux vidéo en direct en ligne.
