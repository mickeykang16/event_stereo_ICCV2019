docker run -it\
    --name mvsec_convert\
    --shm-size=999G \
    -v ./:/home/user/ws/event_stereo_ICCV2019 \
    -v /home/user/jaeyoung/data/mvsec_data:/root/mvsec_data \
    mvsec_convert:test bash