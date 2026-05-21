# Integrations

Third-party integrations for cloud storage, streaming infrastructure, and dataset labeling.

---

## Cloud Storage

> **Extra required:** `pip install "ai-vision-tool[cloud]"`

### S3Source

Read images directly from an S3 bucket.

```python
from ai_vision_tool.integrations.cloud import S3Source

source = S3Source(
    bucket="my-dataset-bucket",
    prefix="images/train/",
    region="us-east-1",
    # Credentials via standard boto3 env vars or AWS config
)

for key, image in source:
    result = pipeline.execute(image)
```

### GCSSource

Read images from Google Cloud Storage.

```python
from ai_vision_tool.integrations.cloud import GCSSource

source = GCSSource(
    bucket="my-gcs-bucket",
    prefix="dataset/raw/",
    credentials_path="/path/to/service-account.json",
)

for blob_name, image in source:
    result = pipeline.execute(image)
```

---

## WebSocket Streaming

> **Extra required:** `pip install "ai-vision-tool[websocket]"`

### WebSocketSink

Push processed frames to a WebSocket server in real time.

```python
from ai_vision_tool.integrations.streaming import WebSocketSink

sink = WebSocketSink(
    uri="ws://localhost:8765",
    encode_format="jpg",
    encode_quality=80,
)

sink.setup({})
for frame in camera_stream:
    payload = pipeline.execute(frame)
    sink.run(payload)
```

### WebSocketSource

Receive frames from a WebSocket server.

```python
from ai_vision_tool.integrations.streaming import WebSocketSource

source = WebSocketSource(uri="ws://frame-server:8765")
source.setup({})

for payload in source:
    result = pipeline.execute(payload["frame"])
```

---

## Kafka Streaming

> **Extra required:** `pip install "ai-vision-tool[kafka]"`

### KafkaSource

Consume frame messages from a Kafka topic.

```python
from ai_vision_tool.integrations.streaming import KafkaSource

source = KafkaSource(
    bootstrap_servers="kafka:9092",
    topic="raw-frames",
    group_id="vision-consumer",
    auto_offset_reset="latest",
)

source.setup({})
for payload in source:
    result = pipeline.execute(payload["frame"])
```

### KafkaSink

Publish processed frames to a Kafka topic.

```python
from ai_vision_tool.integrations.streaming import KafkaSink

sink = KafkaSink(
    bootstrap_servers="kafka:9092",
    topic="processed-frames",
)

sink.setup({})
for frame in frames:
    payload = pipeline.execute(frame)
    sink.run(payload)
```

---

## Auto-Labeling

Generate YOLO, Darknet, or TensorFlow annotation files automatically.

### AutoLabeller

```python
from ai_vision_tool.integrations.labeling import AutoLabeller
from ai_vision_tool.detection import ObjectDetector

detector = ObjectDetector(model="yolov8n.pt", confidence=0.5)

labeller = AutoLabeller(
    detector=detector,
    output_dir="./labels",
    format="yolo",
    class_names=["cat", "dog", "person"],
)

labeller.label_directory("./images/")
```

### DarknetAutoLabeler

```python
from ai_vision_tool.integrations.labeling import DarknetAutoLabeler

labeler = DarknetAutoLabeler(
    detector=detector,
    output_dir="./darknet_labels",
    class_names=["cat", "dog"],
)
labeler.label_directory("./images/")
```

### TensorFlowAutoLabeler

```python
from ai_vision_tool.integrations.labeling import TensorFlowAutoLabeler

labeler = TensorFlowAutoLabeler(
    detector=detector,
    output_dir="./tf_records",
    class_names=["cat", "dog"],
)
labeler.label_directory("./images/")
```
