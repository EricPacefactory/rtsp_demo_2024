

echo ""
echo "Downloading model files (~600MB total)"


pose_models_path="models/pose"
echo ""
echo "Downloading nano pose model"
wget -P $pose_models_path --no-clobber \
	https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt

echo ""
echo "Downloading small pose model"
wget -P $pose_models_path --no-clobber \
	https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt

echo ""
echo "Downloading medium pose model"
wget -P $pose_models_path --no-clobber \
	https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt


depth_models_path="models/depth"
echo ""
echo "Downloading small depth model"
wget -P $depth_models_path --no-clobber \
	https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vits14.onnx

echo ""
echo "Downloading base depth model"
wget -P $depth_models_path --no-clobber \
	https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vitb14.onnx

