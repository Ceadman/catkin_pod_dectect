import cv2, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)
pipeline = Gst.parse_launch(
    "rtspsrc location=rtsp://192.168.144.119:554 latency=0 buffer-mode=0 drop-on-latency=true protocols=tcp ! "
    "rtph264depay ! h264parse ! "
    "nvv4l2decoder enable-frame-type-reporting=true disable-dpb=true ! "
    "nvvidconv ! "
    "video/x-raw,format=NV12 ! "
    "appsink max-buffers=1 drop=true sync=false emit-signals=true name=sink"
)
sink = pipeline.get_by_name('sink')
sink.connect('new-sample', lambda appsink: cv2.cuda_GpuMat(appsink.pull_sample().get_buffer().extract_dup(0, -1)).shape[0] and True)
pipeline.set_state(Gst.State.PLAYING)