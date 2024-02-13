conda activate autoawq

export https_proxy=http://127.0.0.1:15777 http_proxy=http://127.0.0.1:15777

git config --global http.proxy http://127.0.0.1:15777 && git config --global https.proxy http://127.0.0.1:15777

cd ~/AutoAWQ