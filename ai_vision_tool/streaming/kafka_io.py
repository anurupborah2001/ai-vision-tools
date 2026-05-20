from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame


def _import_consumer():
    try:
        from confluent_kafka import Consumer
        return Consumer, "confluent"
    except ImportError:
        pass
    try:
        from kafka import KafkaConsumer
        return KafkaConsumer, "kafka_python"
    except ImportError:
        raise ImportError("Install with: pip install confluent-kafka  # or kafka-python")


def _import_producer():
    try:
        from confluent_kafka import Producer
        return Producer, "confluent"
    except ImportError:
        pass
    try:
        from kafka import KafkaProducer
        return KafkaProducer, "kafka_python"
    except ImportError:
        raise ImportError("Install with: pip install confluent-kafka  # or kafka-python")


class KafkaSource(AIVisionComponent):
    """Reads frames from a Kafka topic.

    Args:
        bootstrap_servers: Kafka bootstrap server(s).
        topic: Topic to consume from.
        group_id: Consumer group ID.
        auto_offset_reset: "latest" or "earliest".
        timeout_ms: Poll timeout in milliseconds.
    """

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "ai_vision",
                 auto_offset_reset: str = "latest", timeout_ms: int = 1000):
        super().__init__()
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.timeout_ms = timeout_ms
        self._consumer = None
        self._lib = ""

    def setup(self, config: dict):
        cls, self._lib = _import_consumer()
        if self._lib == "confluent":
            self._consumer = cls({
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": self.group_id,
                "auto.offset.reset": self.auto_offset_reset,
            })
            self._consumer.subscribe([self.topic])
        else:
            self._consumer = cls(
                self.topic, bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id, auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda v: v,
            )
        super().setup(config)

    def _execute(self, data, config):
        if self._lib == "confluent":
            msg = self._consumer.poll(self.timeout_ms / 1000.0)
            if msg is None or msg.error():
                return {"frame": None, "connected": False}
            raw = msg.value()
            offset, partition = msg.offset(), msg.partition()
        else:
            record = next(iter(self._consumer), None)
            if record is None:
                return {"frame": None, "connected": False}
            raw = record.value
            offset, partition = record.offset, record.partition

        try:
            d = json.loads(raw)
            buf = np.frombuffer(base64.b64decode(d["frame"]), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return {"frame": frame, "kafka_offset": offset, "kafka_partition": partition,
                    "kafka_topic": self.topic, "connected": True,
                    "metadata": d.get("metadata", {})}
        except Exception as e:
            return {"frame": None, "error": str(e), "connected": False}

    def cleanup(self):
        if self._consumer:
            self._consumer.close()


class KafkaSink(AIVisionComponent):
    """Sends encoded frames to a Kafka topic.

    Args:
        bootstrap_servers: Kafka bootstrap server(s).
        topic: Topic to produce to.
        quality: JPEG encoding quality.
    """

    def __init__(self, bootstrap_servers: str, topic: str, quality: int = 80):
        super().__init__()
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.quality = quality
        self._producer = None
        self._lib = ""

    def setup(self, config: dict):
        cls, self._lib = _import_producer()
        if self._lib == "confluent":
            self._producer = cls({"bootstrap.servers": self.bootstrap_servers})
        else:
            self._producer = cls(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v,
            )
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        payload = json.dumps({
            "frame": base64.b64encode(buf.tobytes()).decode(),
            "timestamp_ms": time.time() * 1000.0,
            "metadata": {},
        }).encode()

        if self._lib == "confluent":
            self._producer.produce(self.topic, value=payload)
            self._producer.poll(0)
        else:
            self._producer.send(self.topic, value=payload)
        return data

    def cleanup(self):
        if self._producer:
            if self._lib == "confluent":
                self._producer.flush()
            else:
                self._producer.close()
