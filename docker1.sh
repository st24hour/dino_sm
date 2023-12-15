docker run --gpus all --shm-size=256g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --pid=host -v $(pwd):$(pwd) -w $(pwd) \
	-v /etc/localtime:/etc/localtime:ro -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro --name js_9 \
	-v /shared:/shared/ st24hour/pytorch:1.12.1_SM \
	sh train1.sh

