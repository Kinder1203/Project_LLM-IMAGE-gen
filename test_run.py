import os
import sys
import logging
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4
import requests
from loguru import logger as loguru_logger
from src.llm_pipeline.pipelines import process_generation_request
from src.llm_pipeline.core.schemas import PipelineRequest
from src.llm_pipeline.core.config import config


def configure_demo_logging():
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
    """ 로컬 이미지를 ComfyUI 서버로 업로드하고, 내부 API용 파일명을 반환합니다. """
    url = f"{config.COMFYUI_URL}/upload/image"
    try:
        with open(local_filepath, "rb") as f:
            files = {"image": f}
            res = requests.post(url, files=files, timeout=config.COMFYUI_UPLOAD_TIMEOUT_SECONDS)
            res.raise_for_status()
            return res.json()["name"]
    except Exception as e:
        print(f"[ERROR] ComfyUI 이미지 업로드 실패. ComfyUI가 켜져있는지 확인하세요: {e}")
        return ""

def download_image(url: str, save_dir: str, prefix: str="output"):
    """ ComfyUI 서버(또는 로컬)에 떠있는 이미지를 현재 PC/Runpod의 폴더로 다운로드합니다 """
    os.makedirs(save_dir, exist_ok=True)
    try:
        res = requests.get(url, timeout=config.IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
        res.raise_for_status()
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        original_filename = query.get("filename", ["output.png"])[0]
        original_filename = unquote(original_filename)
        filepath = os.path.join(save_dir, f"{prefix}_{original_filename}")
        
        with open(filepath, "wb") as f:
            f.write(res.content)
        print(f"[OK] 결과 저장 완료 -> {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] 다운로드 실패. URL: {url} | 에러: {e}")
        return None

def get_input_image_path() -> str:
    filename = input("input_images 폴더 안에 넣은 원본 시안 파일명을 적어주세요 (ex: my_ring.png): ").strip()
    if not filename:
        print("[ERROR] 파일명이 비어있습니다. 메인 메뉴로 돌아갑니다.")
        return ""
        
    filepath = os.path.join("input_images", filename)
    if not os.path.exists(filepath):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {filepath}")
        return ""
    
    return filepath


def review_customized_design(thread_id: str, response, download_prefix: str):
    current_response = response

    while current_response.status == "waiting_for_user_edit":
        print("\n[휴게소 도달] 커스텀 결과 확인 대기")
        custom_url = current_response.base_image_url
        if custom_url:
            download_image(custom_url, "output_images", download_prefix)

        custom_choice = input(
            "\n[시안 확인] 해당 커스텀 수정본을 수락하시겠습니까?\n"
            "  1. 수락하고 다각도로!\n"
            "  2. 재수정\n"
            "번호 입력: "
        ).strip()

        if custom_choice == "2":
            redo_prompt = input("\n[재수정 프롬프트] 어떻게 다시 바꾸고 싶으신가요?: ").strip()
            if not redo_prompt:
                redo_prompt = "다른 느낌으로 다시 뽑아봐"
            print("\n유저 피드백 반영: 커스텀 재수정 요청...\n")
            current_response = process_generation_request(
                PipelineRequest(
                    thread_id=thread_id,
                    action="request_customization",
                    customization_prompt=redo_prompt,
                )
            )
            continue

        print("\n[다음 단계] 현재 시안을 수락하고 다각도 진행...\n")
        current_response = process_generation_request(
            PipelineRequest(thread_id=thread_id, action="accept_base")
        )

    return current_response

def run_interactive_demo():
    configure_demo_logging()
    # 데모 실행 중에는 외부 백엔드로 웹훅을 보내지 않는다.
    config.WEBHOOK_URL = "NONE"
    os.makedirs("output_images", exist_ok=True)
    os.makedirs("input_images", exist_ok=True)
    
    while True:
        print("\n" + "="*70)
        print("[반지 커스텀 파이프라인 졸업작품 종합 시연(Demo) 메뉴]")
        print("*" * 70)
        print(" 1. [완전 커스텀]   백지 프롬프트 -> 1차 시안 -> 사용자 검토 -> 다각도/커스텀")
        print(" 2. [이미지 기반 수정] 준비된 시안 이미지 업로드 -> 텍스트로 각인/수정 -> 다각도 완성")
        print(" 3. [다각도 즉시 추출] 완성된 반지 이미지 업로드 -> 즉시 다각도/누끼 추출")
        print(" 4. [종료] 프로그램 끄기")
        print("-" * 70)
        
        mode = input("원하시는 시연 분기의 번호를 입력하세요: ").strip()
        
        if mode == "1":
            # ======================== [시나리오 1 - 텍스트 온리] ========================
            thread_id = f"demo_scenario_1_{uuid4().hex[:8]}"
            user_prompt = input("\n[프롬프트] 어떤 반지를 만들고 싶으신가요?: ")
            if not user_prompt: user_prompt = "18k 화이트골드 모던 커플링"
                
            print(f"\n[Step 1] 베이스 링 생성을 시작합니다 (입력값: {user_prompt})...\n")
            request_start = PipelineRequest(
                thread_id=thread_id, action="start", input_type="text", prompt=user_prompt
            )
            res_start = process_generation_request(request_start)
            
            if res_start.status == "waiting_for_user":
                print("\n[1단계 휴게소 도달] 일시정지 (Human-in-the-loop 작동)")
                base_url = res_start.base_image_url
                if base_url:
                    download_image(base_url, "output_images", "scenario1_base")
                    
                print("\n---------------------------------------------------------")
                choice = input("[분기 선택] 생성된 시안을 확인했습니다.\n  1. 완벽해! 다각도/누끼 바로 떠줘\n  2. 각인/보석 같은 커스텀 옵션 추가\n번호 입력: ")
                
                if choice.strip() == "2":
                    custom_prompt = input("\n[추가 프롬프트] 어떤 커스텀을 원하시나요?: ")
                    print("\n[Step 2] 유저 피드백 수용. 커스텀 수정 파이프라인 재가동...\n")
                    request_custom = PipelineRequest(
                        thread_id=thread_id, action="request_customization", customization_prompt=custom_prompt
                    )
                    res_custom = process_generation_request(request_custom)
                    
                    if res_custom.status == "waiting_for_user_edit":
                        res_final = review_customized_design(
                            thread_id=thread_id,
                            response=res_custom,
                            download_prefix="scenario1_custom_review",
                        )
                    else:
                        res_final = res_custom
                else:
                    print("\n[Step 2] 베이스 수락 완료. 다각도 투명화 모듈 직행...\n")
                    request_accept = PipelineRequest(thread_id=thread_id, action="accept_base")
                    res_final = process_generation_request(request_accept)
                    
                if res_final.status == "success":
                    print("\n[최종 완료] 3D TRELLIS용 다각도/누끼 처리 완료!")
                    for idx, url in enumerate(res_final.optimized_image_urls): download_image(url, "output_images", f"scenario1_final_{idx}")
                else: print(f"[ERROR] {res_final.message}")
            else:
                print(f"[ERROR] {res_start.message}")

        elif mode == "2":
            # ======================== [시나리오 2 - 이미지 + 프롬프트] ========================
            thread_id = f"demo_scenario_2_{uuid4().hex[:8]}"
            print("\n이미지를 수정합니다. 'input_images' 폴더에 이미지를 미리 넣어주세요.")
            filepath = get_input_image_path()
            if not filepath: continue
            
            comfy_img_name = upload_image_to_comfyui(filepath)
            if not comfy_img_name: continue
                
            custom_prompt = input("\n[수정 프롬프트] 해당 시안에 어떤 커스텀(각인/큐빅)을 더하시겠습니까?: ")
            if not custom_prompt: custom_prompt = "반지 중앙에 거대한 사파이어 합성"
            
            print("\n파이프라인 가동 (이미지 업로드 + 커스텀 -> 모델 검수 -> 수정본 확인 -> 다각도 진행)...\n")
            request = PipelineRequest(
                thread_id=thread_id, action="start", input_type="image_and_text", 
                prompt=custom_prompt, image_url=comfy_img_name
            )
            res = process_generation_request(request)
            
            if res.status == "waiting_for_user_edit":
                res_final = review_customized_design(
                    thread_id=thread_id,
                    response=res,
                    download_prefix="scenario2_custom_review",
                )
            else:
                res_final = res

            if res_final.status == "success":
                    print("\n[최종 완료] 이미지 커스텀 완료 및 다각도 추출 누끼 완료!")
                    for idx, url in enumerate(res_final.optimized_image_urls): download_image(url, "output_images", f"scenario2_custom_final_{idx}")
            else: print(f"[ERROR] {res_final.message}")

        elif mode == "3":
            # ======================== [시나리오 3 - 다각도 온리] ========================
            thread_id = f"demo_scenario_3_{uuid4().hex[:8]}"
            print("\n완성된 반지의 다각도 분할/누끼만 처리합니다. 'input_images' 폴더에 이미지를 미리 넣어주세요.")
            filepath = get_input_image_path()
            if not filepath: continue
            
            comfy_img_name = upload_image_to_comfyui(filepath)
            if not comfy_img_name: continue
            
            print("\n파이프라인 가동 (다각도 분해 모듈 직행)...\n")
            # 프롬프트 없이 시안 URL만 넘기면 파이프라인이 알아서 multi_view_only 인텐트로 간주합니다.
            request = PipelineRequest(
                thread_id=thread_id, action="start", input_type="image_only", 
                image_url=comfy_img_name
            )
            res = process_generation_request(request)
            
            if res.status == "success":
                print("\n[최종 완료] 원본 이미지 다각도 추출 및 TRELLIS용 배경 투명화 완료!")
                for idx, url in enumerate(res.optimized_image_urls): download_image(url, "output_images", f"scenario3_rembg_final_{idx}")
            else: print(f"[ERROR] {res.message}")

        elif mode == "4":
            print("\n시연 프로그램을 종료합니다.")
            break
        else:
            print("\n[ERROR] 1 ~ 4 사이의 숫자를 입력해주세요.")

if __name__ == "__main__":
    run_interactive_demo()
