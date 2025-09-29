## How to use this whisper
+ connect to CCU school wifi or VPN
+ use linux env (other ask gpt)
+ input the command below with filename changed to yours

```bash
curl -X POST http://140.123.105.233:8080/transcribe/ -H "accept: application/json" -F "files=@<<file _ name + file type >>" -o output.txt
```
+ get the audio file to text in output.txt

## deploy 
+ mend only whisper_api.py
+ redeploy as below
```bash
uvicorn whisper_api:app --host 0.0.0.0 --port 8080
```

## requirement
+ pip install -r requirements.txt
+ install ffmpeg
```bash
# goto ~/install_app
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# unzip
tar -xf ffmpeg-release-amd64-static.tar.xz

# see ffmpeg-*-amd64-static/ in pwd
# add to PATH
cd ffmpeg-*-static
mkdir -p ~/bin
cp ffmpeg ~/bin/
cp ffprobe ~/bin/

# mend .bashrc
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# see if really work
which ffmpeg
ffmpeg -version

```