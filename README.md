# RTSP Demo 2024

This is a script built for a demo in April 2024. It can run [YOLO pose](https://github.com/ultralytics/ultralytics) models, [ArUco marker](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) detection and [depth prediction](https://depth-anything.github.io/) models on live video (e.g. from RTSP sources).
The script also supports running off a webcam feed, a video file or even static images! It may be useful for quickly verifying performance of models on sample videos.

<p align="center">
  <img src=".readme_assets/demo_image.webp">
</p>


## Install

To run the script, you first need to create a virtual environment and install some dependencies. This can be done as follows:

```bash
# Linux & MacOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```


## Running the script

To run the script, make sure the virtual environment is activated (the one created in the [Install](https://github.com/EricPacefactory/rtsp_demo_2024?tab=readme-ov-file#install) section), and then run:

```bash
python demo.py
```

Optionally there is a `-i` flag that can be used to specify the input video source and a `-s` flag which can be used to adjust the initial display sizing (the up/down arrow keys can be used to adjust the sizing while running).
Input sources can take the form of:
- an rtsp url (e.g. `rtsp://username:password@123.45.67.89:554/stream`)
- a path to a video file (e.g. `/path/to/video.mp4`)
- a path to an image (e.g. `/path/to/picture.jpg`)
- or an integer can be given, which will select a webcam if available (0 selects the 'first' webcam, 1, 2, 3 etc. can be used to read from other webcams if more than one is present)

If the `-i` flag is not provided, you'll be asked to enter the input source when running the script.

### Helper scripts

For convenience, there are helper scripts available which handle all of the setup (including creating/activating the virtual environment) and runs the script. This can be used as follows:

```bash
# Linux & MacOS
./run.sh

# Windows (cmd)
winrun.bat
```

The `-i` and `-s` flags can also be passed to these scripts.


## Model downloads

For the script to work, there must be at least one pose model as well as one depth model on the system. The demo script will try to download these automatically if they're not found on startup.

If you'd like to download these manually, the YOLO pose models can be downloaded frm the [Ultralytics page](https://docs.ultralytics.com/tasks/pose/). Depth models must be in onnx format, which can be downloaded from the [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX/releases) github page. No download is needed for ArUco markers, though you will need to [generate](https://chev.me/arucogen/) valid ArUco patterns.

