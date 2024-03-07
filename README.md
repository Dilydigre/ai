# Description

Dépôt github pour l'ia et l'API

# Setup

- La première chose à faire est de se créer une clé SSH et l'ajouter sur github (ou alors d'utiliser la sienne déjà générée) si ce n'a pas encore été fait.
La procédure est disponible [ici](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
- Cloner le dépôt en local `git clone git@github.com:Dilydigre/ai.git`
- Se placer sur la branche de dev `git checkout dev/<ai ou api>` pour le développement (on ne commit que sur la *main* quand tout fonctionne correctement et que ce que l'on est en train de développer est fini)
- Créer un [venv](https://docs.python.org/3/library/venv.html) avec `python3 -m venv venv`
- Utiliser ce venv : `source venv/bin/activate` et si besoin installer les modules avec `pip install -r requirements.txt`

# Développement
- Penser à utiliser le venv quand on développe et mettre à jour le fichier requirements.txt quand on installe un nouveau module pour le projet avec `pip freeze > requirements.txt`
- La liste des tâches à faire est dans [TODO.md](TODO.md)
- Rédiger la documentation dans [DOC.md](DOC.md)