# 基于的基础镜像
FROM python:3.6

# 代码添加到code文件夹
ADD ./multilabel /code

# 设置code文件夹是工作目录
WORKDIR /code

# 安装支持
RUN pip install -r requirements.txt

EXPOSE 8000
EXPOSE 9000

CMD ["/bin/bash", "main.sh"]


