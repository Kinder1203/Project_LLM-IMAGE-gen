import logging
import os
import sys
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import requests
from loguru import logger as loguru_logger

from src.core.config import config
from src.core.schemas import PipelineRequest
from src.pipelines import process_generation_request


def configure_demo_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )


def upload_image_to_comfyui(local_filepath: str) -> str:
    url = f"{config.COMFYUI_URL}/upload/image"
    try:
        with open(local_filepath, "rb") as file_handle:
            files = {"image": file_handle}
            response = requests.post(url, files=files, timeout=config.COMFYUI_UPLOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()["name"]
    except Exception as exc:
        print(f"[ERROR] ComfyUI 이미지 업로드 실패: {exc}")
        return ""


def download_image(url: str, save_dir: str, prefix: str = "output") -> str | None:
    os.makedirs(save_dir, exist_ok=True)
    try:
        response = requests.get(url, timeout=config.IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        original_filename = query.get("filename", ["output.png"])[0]
        original_filename = unquote(original_filename)
        filepath = os.path.join(save_dir, f"{prefix}_{original_filename}")

        with open(filepath, "wb") as file_handle:
            file_handle.write(response.content)
        print(f"[OK] 결과 저장 완료 -> {filepath}")
        return filepath
    except Exception as exc:
        print(f"[ERROR] 이미지 다운로드 실패. URL: {url} | 오류: {exc}")
        return None


def get_input_image_path(prompt_text: str) -> str:
    filename = input(prompt_text).strip()
    if not filename:
        return ""

    filepath = os.path.join("input_images", filename)
    if not os.path.exists(filepath):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {filepath}")
        return ""

    return filepath


def _upload_optional_reference(label: str) -> str | None:
    filepath = get_input_image_path(
        f"{label} 참고 이미지를 쓰려면 input_images 폴더 안 파일명을 입력하세요. 없으면 Enter: "
    )
    if not filepath:
        return None

    uploaded_name = upload_image_to_comfyui(filepath)
    if not uploaded_name:
        print(f"[WARN] {label} 참고 이미지 업로드에 실패해 이번 요청에서는 제외합니다.")
        return None
    return uploaded_name


def collect_optional_reference_images() -> dict[str, str | None]:
    engraving_ref = _upload_optional_reference("각인/문양")
    gemstone_ref = _upload_optional_reference("큐빅/보석")
    return {
        "engraving_reference_image_url": engraving_ref,
        "gemstone_reference_image_url": gemstone_ref,
    }


def review_customized_design(thread_id: str, response, download_prefix: str):
    current_response = response

    while current_response.status == "waiting_for_user_edit":
        print("\n[검토 단계] 커스터마이즈 결과 확인 대기")
        custom_url = current_response.base_image_url
        if custom_url:
            download_image(custom_url, "output_images", download_prefix)

        custom_choice = input(
            "\n[시안 확인] 현재 커스터마이즈 결과를 승인할까요?\n"
            "  1. 승인하고 다각도 생성 진행\n"
            "  2. 다시 수정 요청\n"
            "번호 입력: "
        ).strip()

        if custom_choice == "2":
            redo_prompt = input("\n[수정 프롬프트] 어떻게 다시 바꾸고 싶은지 입력하세요: ").strip()
            if not redo_prompt:
                redo_prompt = "좀 더 자연스럽게 다시 수정"
            reference_kwargs = collect_optional_reference_images()
            print("\n[재요청] 수정 피드백을 반영해 다시 편집합니다...\n")
            current_response = process_generation_request(
                PipelineRequest(
                    thread_id=thread_id,
                    action="request_customization",
                    customization_prompt=redo_prompt,
                    engraving_reference_image_url=reference_kwargs["engraving_reference_image_url"],
                    gemstone_reference_image_url=reference_kwargs["gemstone_reference_image_url"],
                )
            )
            continue

        print("\n[다음 단계] 현재 시안을 승인하고 다각도 생성으로 진행합니다...\n")
        current_response = process_generation_request(
            PipelineRequest(thread_id=thread_id, action="accept_base")
        )

    return current_response


def run_interactive_demo() -> None:
    configure_demo_logging()
    config.WEBHOOK_URL = "NONE"
    os.makedirs("output_images", exist_ok=True)
    os.makedirs("input_images", exist_ok=True)

    while True:
        print("\n" + "=" * 70)
        print("[반지 커스터마이즈 파이프라인 졸작 데모]")
        print("*" * 70)
        print(" 1. [텍스트 기반 생성] 프롬프트로 기본 반지 생성 후 승인 또는 수정")
        print(" 2. [이미지 + 프롬프트 편집] 기존 반지 이미지를 업로드하고 수정")
        print(" 3. [이미지 즉시 다각도] 완성 반지 이미지를 업로드하고 바로 다각도 추출")
        print(" 4. [종료]")
        print("-" * 70)

        mode = input("실행할 시나리오 번호를 입력하세요: ").strip()

        if mode == "1":
            thread_id = f"demo_scenario_1_{uuid4().hex[:8]}"
            user_prompt = input("\n[프롬프트] 어떤 반지를 만들고 싶은가요?: ").strip()
            if not user_prompt:
                user_prompt = "18k 화이트 골드 커플링, 미니멀한 제품 사진"

            print(f"\n[Step 1] 기본 반지 생성을 시작합니다. 입력: {user_prompt}\n")
            request_start = PipelineRequest(
                thread_id=thread_id,
                action="start",
                input_type="text",
                prompt=user_prompt,
            )
            res_start = process_generation_request(request_start)

            if res_start.status != "waiting_for_user":
                print(f"[ERROR] {res_start.message}")
                continue

            print("\n[1차 검토] 기본 시안이 도착했습니다.")
            if res_start.base_image_url:
                download_image(res_start.base_image_url, "output_images", "scenario1_base")

            choice = input(
                "\n[분기 선택]\n"
                "  1. 현재 시안을 승인하고 다각도 생성 진행\n"
                "  2. 각인/보석 커스터마이즈 추가\n"
                "번호 입력: "
            ).strip()

            if choice == "2":
                custom_prompt = input("\n[추가 프롬프트] 어떤 커스터마이즈를 원하나요?: ").strip()
                reference_kwargs = collect_optional_reference_images()
                print("\n[Step 2] 커스터마이즈 편집을 진행합니다...\n")
                res_custom = process_generation_request(
                    PipelineRequest(
                        thread_id=thread_id,
                        action="request_customization",
                        customization_prompt=custom_prompt or "더 정교한 각인과 포인트 스톤 추가",
                        engraving_reference_image_url=reference_kwargs["engraving_reference_image_url"],
                        gemstone_reference_image_url=reference_kwargs["gemstone_reference_image_url"],
                    )
                )
                res_final = (
                    review_customized_design(
                        thread_id=thread_id,
                        response=res_custom,
                        download_prefix="scenario1_custom_review",
                    )
                    if res_custom.status == "waiting_for_user_edit"
                    else res_custom
                )
            else:
                print("\n[Step 2] 기본 시안을 승인하고 다각도 생성을 진행합니다...\n")
                res_final = process_generation_request(
                    PipelineRequest(thread_id=thread_id, action="accept_base")
                )

            if res_final.status == "success":
                print("\n[최종 완료] 다각도 생성 완료")
                for idx, url in enumerate(res_final.optimized_image_urls):
                    download_image(url, "output_images", f"scenario1_final_{idx}")
            else:
                print(f"[ERROR] {res_final.message}")

        elif mode == "2":
            thread_id = f"demo_scenario_2_{uuid4().hex[:8]}"
            print("\ninput_images 폴더 안에 원본 반지 이미지를 먼저 넣어두세요.")
            filepath = get_input_image_path("원본 반지 파일명을 입력하세요 (예: my_ring.png): ")
            if not filepath:
                continue

            comfy_img_name = upload_image_to_comfyui(filepath)
            if not comfy_img_name:
                continue

            custom_prompt = input("\n[수정 프롬프트] 어떤 편집을 원하는가요?: ").strip()
            if not custom_prompt:
                custom_prompt = "반지 중앙에 작은 사파이어를 추가"

            reference_kwargs = collect_optional_reference_images()
            print("\n[Step 1] 이미지 편집 파이프라인을 시작합니다...\n")
            request = PipelineRequest(
                thread_id=thread_id,
                action="start",
                input_type="image_and_text",
                prompt=custom_prompt,
                image_url=comfy_img_name,
                engraving_reference_image_url=reference_kwargs["engraving_reference_image_url"],
                gemstone_reference_image_url=reference_kwargs["gemstone_reference_image_url"],
            )
            res = process_generation_request(request)
            res_final = (
                review_customized_design(
                    thread_id=thread_id,
                    response=res,
                    download_prefix="scenario2_custom_review",
                )
                if res.status == "waiting_for_user_edit"
                else res
            )

            if res_final.status == "success":
                print("\n[최종 완료] 이미지 편집 및 다각도 추출 완료")
                for idx, url in enumerate(res_final.optimized_image_urls):
                    download_image(url, "output_images", f"scenario2_final_{idx}")
            else:
                print(f"[ERROR] {res_final.message}")

        elif mode == "3":
            thread_id = f"demo_scenario_3_{uuid4().hex[:8]}"
            print("\ninput_images 폴더 안에 완성 반지 이미지를 먼저 넣어두세요.")
            filepath = get_input_image_path("완성 반지 파일명을 입력하세요 (예: final_ring.png): ")
            if not filepath:
                continue

            comfy_img_name = upload_image_to_comfyui(filepath)
            if not comfy_img_name:
                continue

            print("\n[Step 1] 다각도 추출만 진행합니다...\n")
            request = PipelineRequest(
                thread_id=thread_id,
                action="start",
                input_type="image_only",
                image_url=comfy_img_name,
            )
            res = process_generation_request(request)

            if res.status == "success":
                print("\n[최종 완료] 다각도 추출 완료")
                for idx, url in enumerate(res.optimized_image_urls):
                    download_image(url, "output_images", f"scenario3_final_{idx}")
            else:
                print(f"[ERROR] {res.message}")

        elif mode == "4":
            print("\n데모를 종료합니다.")
            break
        else:
            print("\n[ERROR] 1 ~ 4 사이의 숫자를 입력하세요.")


if __name__ == "__main__":
    run_interactive_demo()
