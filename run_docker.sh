docker run --gpus all \
    --shm-size=16g \
    -it \
    --name scube_container \
    -v /home/xingyu/gs_ws:/home/xingyu/gs_ws \
    -w /home/xingyu/gs_ws/src/SemCity \
    scube:v1 \
    bash -c "cd /home/xingyu/gs_ws/src/SemCity && pip install -e . && pip install blobfile matplotlib prettytable scikit-learn tqdm && bash"\

