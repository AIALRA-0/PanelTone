"""Exercise the local 8781 control contract without touching a PanelTone job.

The script is intentionally separate from the maintenance restart.  It only
uses a synthetic image, records evidence, and fails closed when an assertion
cannot be observed rather than treating a missing control as success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", default="http://127.0.0.1:8781")
    parser.add_argument("--app-url", default="http://127.0.0.1:8765")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--idle-wait-seconds", type=float, default=70.0)
    parser.add_argument("--active-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def write_json(directory: Path, name: str, value: Any) -> None:
    (directory / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def gpu_snapshot() -> dict[str, Any]:
    query = (
        "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,"
        "driver_version"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "available": result.returncode == 0,
            "captured_at": time.time(),
            "output": result.stdout.strip() or result.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "captured_at": time.time(), "reason": str(exc)}


def make_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1024, 1024), (224, 224, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 992, 992), outline=(20, 20, 20), width=8)
    draw.rectangle((96, 96, 480, 520), fill=(246, 201, 177), outline=(30, 30, 30), width=6)
    draw.rectangle((544, 96, 928, 520), fill=(98, 132, 184), outline=(30, 30, 30), width=6)
    draw.ellipse((210, 180, 366, 336), fill=(239, 189, 161), outline=(20, 20, 20), width=6)
    draw.ellipse((660, 180, 816, 336), fill=(239, 189, 161), outline=(20, 20, 20), width=6)
    draw.line((120, 650, 900, 650), fill=(20, 20, 20), width=8)
    draw.text((120, 730), "PANELTONE CONTROL FIXTURE", fill=(20, 20, 20))
    image.save(path, format="PNG")


def get_health(client: httpx.Client, model_url: str) -> dict[str, Any]:
    response = client.get(f"{model_url}/health")
    response.raise_for_status()
    return response.json()


def assert_control_health(health: dict[str, Any], idle_seconds: float) -> None:
    required = {
        "active_requests",
        "last_activity",
        "idle_release_seconds",
        "supports_interrupt",
        "supports_release",
        "state",
        "loaded",
    }
    missing = sorted(required - health.keys())
    if missing:
        raise AssertionError(f"/health 缺少控制字段: {', '.join(missing)}")
    if float(health["idle_release_seconds"]) != idle_seconds:
        raise AssertionError(
            f"闲置回收配置为 {health['idle_release_seconds']}，预期 {idle_seconds}"
        )
    if health["supports_interrupt"] is not True or health["supports_release"] is not True:
        raise AssertionError("8781 未声明中断和显存释放能力")


def start_generation(
    executor: ThreadPoolExecutor, client: httpx.Client, model_url: str, fixture: Path
) -> Future[httpx.Response]:
    def request() -> httpx.Response:
        with fixture.open("rb") as stream:
            return client.post(
                f"{model_url}/generate",
                files={"source": (fixture.name, stream, "image/png")},
                data={"prompt": "", "seed": "17"},
                timeout=None,
            )

    return executor.submit(request)


def wait_for_active(
    client: httpx.Client, model_url: str, future: Future[httpx.Response], timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_health(client, model_url)
        if int(last.get("active_requests") or 0) > 0 or last.get("state") == "generating":
            return last
        if future.done():
            break
        time.sleep(0.5)
    raise AssertionError(f"没有观察到可中断的活动请求，最后状态为 {last}")


def wait_for_idle_release(
    client: httpx.Client, model_url: str, idle_seconds: float, wait_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = get_health(client, model_url)
        if (
            int(last.get("active_requests") or 0) == 0
            and last.get("loaded") is False
            and last.get("state") == "idle"
        ):
            return last
        time.sleep(min(2.0, max(0.2, idle_seconds / 10)))
    raise AssertionError(f"等待 {wait_seconds:.0f} 秒后模型仍未自动回收: {last}")


def is_interrupted_response(response: httpx.Response) -> bool:
    if response.status_code == 499:
        return True
    if response.status_code != 500:
        return False
    return any(
        marker in response.text.lower()
        for marker in ("interrupt", "interrupted", "中断")
    )


def main() -> int:
    args = parse_args()
    if args.idle_seconds != 60:
        raise SystemExit("本次验收固定要求 idle_release_seconds=60")
    args.evidence_directory.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture or args.evidence_directory / "synthetic-control-fixture.png"
    if not fixture.is_file():
        make_fixture(fixture)

    with httpx.Client(timeout=20.0) as client, ThreadPoolExecutor(max_workers=2) as executor:
        before = {"health": get_health(client, args.model_url), "gpu": gpu_snapshot()}
        assert_control_health(before["health"], args.idle_seconds)
        write_json(args.evidence_directory, "controls-before.json", before)

        idle_release = client.post(f"{args.model_url}/release")
        if idle_release.status_code != 200 or idle_release.json().get("status") != "released":
            raise AssertionError(
                f"空闲释放未返回 released: {idle_release.status_code} {idle_release.text}"
            )
        released_health = get_health(client, args.model_url)
        if released_health.get("loaded") is not False or released_health.get("state") != "idle":
            raise AssertionError(f"手动释放后健康状态不正确: {released_health}")

        interrupted_response: httpx.Response | None = None
        busy_release_response: httpx.Response | None = None
        for _attempt in range(3):
            future = start_generation(executor, client, args.model_url, fixture)
            active_health = wait_for_active(
                client, args.model_url, future, args.active_timeout_seconds
            )
            busy_release_response = client.post(f"{args.model_url}/release")
            if busy_release_response.status_code != 409:
                future.result(timeout=120)
                continue
            client.post(f"{args.model_url}/interrupt")
            interrupted_response = future.result(timeout=120)
            if is_interrupted_response(interrupted_response):
                break
        if busy_release_response is None or busy_release_response.status_code != 409:
            raise AssertionError("活动请求期间 /release 没有返回 409")
        if interrupted_response is None or not is_interrupted_response(interrupted_response):
            raise AssertionError(
                f"中断请求未返回明确中断状态: {getattr(interrupted_response, 'status_code', None)}"
            )
        after_interrupt = get_health(client, args.model_url)
        if int(after_interrupt.get("active_requests") or 0) != 0:
            raise AssertionError(f"中断后仍有活动请求: {after_interrupt}")

        successful = start_generation(executor, client, args.model_url, fixture).result(timeout=300)
        if successful.status_code != 200:
            raise AssertionError(f"中断后无法继续生成: {successful.status_code} {successful.text}")
        generated_health = get_health(client, args.model_url)
        write_json(
            args.evidence_directory,
            "controls-interrupt.json",
            {
                "active_health": active_health,
                "busy_release": {
                    "status_code": busy_release_response.status_code,
                    "body": busy_release_response.text,
                },
                "interrupt_response": {
                    "status_code": interrupted_response.status_code,
                    "body": interrupted_response.text,
                },
                "after_interrupt": after_interrupt,
                "after_success": generated_health,
                "gpu": gpu_snapshot(),
            },
        )

        auto_released = wait_for_idle_release(
            client, args.model_url, args.idle_seconds, args.idle_wait_seconds
        )
        if args.app_url:
            try:
                logs = client.get(f"{args.app_url}/api/logs?kind=raw&limit=200").json()
            except (httpx.HTTPError, ValueError):
                logs = {"available": False}
        else:
            logs = {"available": False}
        write_json(
            args.evidence_directory,
            "controls-after-idle.json",
            {"health": auto_released, "gpu": gpu_snapshot(), "raw_logs": logs},
        )
        release_markers = ("auto-releas", "自动释放", "自动回收")
        if isinstance(logs, list) and logs and not any(
            any(marker in str(item.get("message", "")).lower() for marker in release_markers)
            for item in logs
        ):
            raise AssertionError("原始日志中未找到自动回收事件")

        final = start_generation(executor, client, args.model_url, fixture).result(timeout=300)
        if final.status_code != 200:
            raise AssertionError(f"自动回收后无法重新加载并生成: {final.status_code} {final.text}")
        write_json(
            args.evidence_directory,
            "controls-final.json",
            {"health": get_health(client, args.model_url), "gpu": gpu_snapshot()},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
