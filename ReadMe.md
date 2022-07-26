# ReadMe

1. Contexte
1. Présentation
2. Structure des fichiers

##  1. Contexte 

​	La connaissance des conditions climatiques en montagne est essentielle mais parfois difficile d’accès. Le but de cette étude est de mettre en place des  méthodes de vision par ordinateur spécialisées pour la tâche de segmentation de la neige. Cela va en effet permettre de valoriser les images issues de webcams déjà présentent dans certains massifs montagneux. On va évaluer aussi bien des méthodes supervisées et non-supervisées telles que le seuillage, le k-means et d’autres. Des résultats sur la classification via différentes méthodes seront aussi présentés. En parallèle de cela, les performances seront constamment évaluées via des métriques usuelles pour rester objectif sur les conclusions apportées. Finalement, au terme de cette étude, on sera en mesure de choisir la méthode de segmentation la plus adéquate pour le cas spécifique de la segmentation de la neige à partir d’images de webcams.

​	Cette étude s'est faite dans le cadre d'un stage de 4ème année de formation ingénieur. Réalisé au sein du laboratoire de recherche [LISTIC ](https://www.univ-smb.fr/listic/) et avec les données du [CREA Mont Blanc](https://creamontblanc.org/fr/).

## 2. Présentation

​	Ce module permet d'expérimenter avec différentes méthodes de segmentation pour le cas spécifique de l'isolement de la neige. Les codes fonctionnent avec des formats d'images classiques. Les datasets utilisés dans les codes sont à recréer en fonction des données disponibles.

​	Dans le répertoire  *exemples_notebooks* des fichiers jupyter notebook permettent d'étudier plus en détail chaque méthode. Un notebook est aussi consacré à l'évaluation des performances des méthodes implémentées. 

​	Les histogrammes de référence (format pickle) utilisées sont aussi disponibles dans le répertoire *data/supervise*  

## 2. Structure des fichiers

<pre><font color="#3465A4"><b>Segmentation_neige</b></font>
├── <font color="#3465A4"><b>codes</b></font>
│   ├── <font color="#4E9A06"><b>DBA.py</b></font>
│   └── <font color="#4E9A06"><b>fonctions.py</b></font>
├── <font color="#3465A4"><b>data</b></font>
│   ├── <font color="#3465A4"><b>Non_supervise</b></font>
│   │   ├── <font color="#3465A4"><b>Imgs</b></font> [175 files]
│   │   └── <font color="#3465A4"><b>Masques</b></font>
│   │       ├── <font color="#3465A4"><b>maskNumpy</b></font> [175 files]
│   │       └── <font color="#3465A4"><b>maskPNG</b></font> [175 files]
│   ├── <font color="#3465A4"><b>Ref</b></font>
│   │   ├── <font color="#3465A4"><b>Imgs</b></font>
│   │   │   ├── loriaz1600Ouest__2018-09-02__13-00-01(1).JPG
│   │   │   ├── loriaz1600ouest__2018-12-15__15-00-00(1).JPG
│   │   │   ├── loriaz1600ouest__2019-04-16__15-00-00(1).JPG
│   │   │   ├── peclerey2000__2018-10-07__13-00-00(1).JPG
│   │   │   ├── peclerey2000__2018-12-12__16-00-01(1).JPG
│   │   │   ├── peclerey2000__2020-04-22__16-00-00(1).JPG
│   │   │   ├── peclerey2400__2019-01-15__13-00-00(1).JPG
│   │   │   ├── peclerey2400__2019-03-24__13-00-01(1).JPG
│   │   │   └── peclerey2400__2019-06-24__13-00-00(1).JPG
│   │   └── <font color="#3465A4"><b>Masques</b></font>
│   │       └── labels_realite_terrain.json
│   ├── <font color="#3465A4"><b>Serie_temporelle_Peclerey_2019</b></font> [356 files]
│   └── <font color="#3465A4"><b>Supervise</b></font>
│       ├── <font color="#3465A4"><b>25</b></font>
│       │   ├── <font color="#3465A4"><b>Avec_neige</b></font> [1962 files]
│       │   └── <font color="#3465A4"><b>Sans_neige</b></font> [1962 files]
│       ├── <font color="#3465A4"><b>5</b></font>
│       │   ├── <font color="#3465A4"><b>Avec_neige</b></font> [3325 files]
│       │   └── <font color="#3465A4"><b>Sans_neige</b></font> [3325 files]
│       ├── <font color="#3465A4"><b>9</b></font>
│       │   ├── <font color="#3465A4"><b>Avec_neige</b></font> [3393 files]
│       │   └── <font color="#3465A4"><b>Sans_neige</b></font> [3393 files]
│       ├── hist_DBA_9.pkl
│       ├── hist_moy_9.pkl
│       └── hist_stack_9.pkl
├── <font color="#3465A4"><b>exemples_notebooks</b></font> [12 files]
│       ├── <font color="#4E9A06"><b>colorSpace.ipynb</b></font>
│       ├── <font color="#4E9A06"><b>evaluationDesPerformances.ipynb</b></font>
│       ├── <font color="#4E9A06"><b>HistClassification.ipynb</b></font>
│       ├── <font color="#4E9A06"><b>K-mean.ipynb</b></font>
│       ├── <font color="#4E9A06"><b>Seuillage.ipynb</b></font>
│       ├── <font color="#4E9A06"><b>SVM.ipynb</b></font>
│       └── <font color="#4E9A06"><b>U-net.ipynb</b></font>
└── <font color="#3465A4"><b>Modèles_trained</b></font>
    ├── <font color="#3465A4"><b>Classif_clusters</b></font>
    │   ├── <font color="#4E9A06"><b>keras_model.h5</b></font>
    │   ├── <font color="#4E9A06"><b>labels.txt</b></font>
    │   └── <font color="#4E9A06"><b>ReadMe.txt</b></font>
    ├── <font color="#3465A4"><b>Classif_imagettes</b></font>
    └── <font color="#3465A4"><b>Segmentation_Unet</b></font>
        └── 20ep_200step_V8.h5
</pre>




