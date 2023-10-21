# D�tection d'un objet statique 

- But : 
D�tection d'un objet qui va entrer dans le champ de la cam�ra et rester statique pendant un certain temps.
Par exemple un sac, une poubelle.... Le programme entourre ces zones en rouge.

- Principe: D�tection des objets statiques puis classification

![Detection du baggage](images/test_image.JPG)
![Detection du baggage](images/test_image2.JPG)

## I. Soustraction de background :

Afin de d�tecter les objets qui restent statique pendant plusieurs secondes, on pratique ce que l'on appelle une "soustraction de fond".
Voici les �tapes de l'algorithme:
1) En entr�e, se trouvent les images correspondant aux frames de la vid�o. On traite les image une par une.
2) L'image est binariz�e 2 fois selon la m�thode de Zivkovich. Diff�rents learning rates sont utilis�espour qu'un des background 
soient mis � jours plus lentement que l'autre ( un long, un court). On obtient alors deux images contenant les objets en mouvement.
3) On effectue un test sur chaque pixel des 2 background. Si celui-ci appartent � un objet en mouvement sur l'image avec les long 
learning rate mais pas � celle avec le court learning rate alors il appartion � un objet qui est arriv� sur l'image et reste imobile pendant 
un certain temps.
4) Si un groupe de plus de N pixels sont c�te � c�te alors on le concid�re comme un objet. On l'entourre alors en rouge.

## II. Classification :
Une fois les objets statiques d�tect�s nous les classifions comme baggages ou non. Pour cela nous avons entrain� un Convolutional Neural Netwark sur des images de sac, valises, train, personnes, .... Pour augmenter la robustesse de notre mod�le nous avons utlis� un r�seau de neurones pr�entrain� pour lequel nous n'avous entrain� que les derni�res couches.

- Test du programme:
1) Choix du Dataset dans le programme main.py
2) On peut modifier les param�tres dans const.py : Les deux learning rates et les deux valeurs de Threshold
2) Lancer main.py en choisisant le dataset voulu pour travailler sur une vid�o, ou lancer main_url.py en indiquant le lien http voulu pour travailler sur un flux vid�o en direct en ligne.

