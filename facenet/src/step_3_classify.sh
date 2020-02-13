set -e

python classifier.py TRAIN /home/lab/zmr/FacialDetection/facenet/imgs/EP05/EP05_playerlist /home/lab/zmr/FacialDetection/facenet/result/EP05/EP05_classifier.pkl

python classifier.py CLASSIFY /home/lab/zmr/FacialDetection/facenet/imgs/EP05/EP05_playerlist /home/lab/zmr/FacialDetection/facenet/result/EP05/EP05_classifier.pkl
