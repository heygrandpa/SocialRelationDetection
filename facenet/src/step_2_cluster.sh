set -e

python contributed/cluster.py /home/lab/zmr/FacialDetection/facenet/imgs/EP09/EP09_drop_similar /home/lab/zmr/FacialDetection/facenet/result/EP09/EP09_cluster_0.6_3 --gpu_memory_fraction 0.9

python contributed/cluster.py /home/lab/zmr/FacialDetection/facenet/imgs/EP10/EP10_drop_similar /home/lab/zmr/FacialDetection/facenet/result/EP10/EP10_cluster_0.6_3 --gpu_memory_fraction 0.9

python contributed/cluster.py /home/lab/zmr/FacialDetection/facenet/imgs/EP11/EP11_drop_similar /home/lab/zmr/FacialDetection/facenet/result/EP11/EP11_cluster_0.6_3 --gpu_memory_fraction 0.9

python contributed/cluster.py /home/lab/zmr/FacialDetection/facenet/imgs/EP12/EP12_drop_similar /home/lab/zmr/FacialDetection/facenet/result/EP12/EP12_cluster_0.6_3 --gpu_memory_fraction 0.9

python contributed/cluster.py /home/lab/zmr/FacialDetection/facenet/imgs/EP13/EP13_drop_similar /home/lab/zmr/FacialDetection/facenet/result/EP13/EP13_cluster_0.6_3 --gpu_memory_fraction 0.9

python contributed/cluster.py /home/lab/zmr/FacialDetection/facenet/imgs/EP14/EP14_drop_similar /home/lab/zmr/FacialDetection/facenet/result/EP14/EP14_cluster_0.6_3 --gpu_memory_fraction 0.9
