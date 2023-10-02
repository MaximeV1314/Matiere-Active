Ce code Python permet de créer une animation d'une dynamique de particules actives.

La première partie du code consiste à placer aléatoirement les particules sans que deux particules soient l'une sur l'autre. La seconde consiste à résoudre l'équation du mouvement d'une particule active dans l'approximation où $M/ \gamma$ est très petit. Les conditions aux limites sont de Dirichlet.

Pour créer une animation, créer dans le même dossier que celui du code python un dossier nommé "img". Exécuter le code. Une fois fini, ouvrir le cmd depuis le dossier initial et taper : ffmpeg -r 30 -i img/%04d.png -vcodec libx264 -y -an video_mq.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
une vidéo .mp4 devrait être créée.

Améliorations prévues :
- créer une interface,
