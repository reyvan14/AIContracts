# 智能合同管理系统
# 部署步骤：
* 1、docker build --no-cache -t reyvan14/chatqna:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .
* 2、docker compose up -d
* 3、docker run -p 8008:80 -v "/Qwen/Qwen2.5-3B-Instruct":/data --name vllm-service --shm-size 128g opea/vllm:latest --model /data --host 0.0.0.0 --port 80
* 4、打开浏览器访问http://0.0.0.0