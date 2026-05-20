import json

import cv2

from ai_vision_tool.cli.main import main


def test_cli_process_image_path_saves_processed_image(sample_frame, tmp_path, capsys):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    ok, encoded = cv2.imencode(".png", sample_frame)
    assert ok is True
    input_path.write_bytes(encoded.tobytes())

    main(
        [
            "--process-image-path",
            "--component-category",
            "augmentation",
            "--component-name",
            "Flip",
            "--image-path",
            str(input_path),
            "--init-args-json",
            '{"horizontal": true}',
            "--save-output-image",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    json_start = captured.out.find("{")
    assert json_start >= 0
    result = json.loads(captured.out[json_start:])

    assert result["category"] == "augmentation"
    assert result["name"] == "Flip"
    assert result["result"]["frame"]["type"] == "image"
    assert output_path.exists()
