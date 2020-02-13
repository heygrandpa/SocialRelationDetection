set -e

python contributed/predict.py /home/lab/zmr/FacialDetection/facenet/imgs/EP01/EP01_multi /home/lab/zmr/FacialDetection/facenet/result/EP01 /home/lab/zmr/FacialDetection/facenet/result/EP01/EP01_classifier.pkl --gpu_memory_fraction 0.9