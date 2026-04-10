import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None
HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("pydantic", "langgraph", "openai")
)
HAS_DB_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("pydantic", "langchain_chroma", "openai")
)


class ExportedTemplateContractTests(unittest.TestCase):
    def test_base_template_is_api_prompt_format(self):
        template = json.loads(Path("image_z_image_turbo (2).json").read_text(encoding="utf-8"))

        self.assertTrue(template)
        self.assertTrue(all("class_type" in node and "inputs" in node for node in template.values()))
        self.assertIn("___USER_PROMPT___", json.dumps(template, ensure_ascii=False))

    def test_edit_template_keeps_exported_load_image_shape(self):
        template = json.loads(Path("image_qwen_image_edit_2509.json").read_text(encoding="utf-8"))
        load_nodes = [node for node in template.values() if node.get("class_type") == "LoadImage"]

        self.assertEqual(len(load_nodes), 1)
        self.assertIn("image", load_nodes[0]["inputs"])
        self.assertNotEqual(load_nodes[0]["inputs"]["image"], "___BASE_IMAGE___")
        self.assertIn("___CUSTOM_PROMPT___", json.dumps(template, ensure_ascii=False))

    def test_multi_view_template_keeps_exported_load_image_shape(self):
        template = json.loads(
            Path("templates-1_click_multiple_character_angles-v1.0 (3) (1).json").read_text(encoding="utf-8")
        )
        load_nodes = [node for node in template.values() if node.get("class_type") == "LoadImage"]
        titled_nodes = [node for node in load_nodes if ((node.get("_meta") or {}).get("title") or "") == "Load Character Image"]

        self.assertGreaterEqual(len(load_nodes), 1)
        self.assertEqual(len(titled_nodes), 1)
        self.assertIn("image", titled_nodes[0]["inputs"])
        self.assertNotEqual(titled_nodes[0]["inputs"]["image"], "___TARGET_IMAGE___")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic is required for schema tests")
class SchemaContractTests(unittest.TestCase):
    def test_input_type_is_normalized_from_payload_shape(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        self.assertEqual(PipelineRequest(thread_id="t1", prompt="ring idea").input_type, "text")
        self.assertEqual(PipelineRequest(thread_id="t2", input_type="image", image_url="sample.png").input_type, "image_only")
        self.assertEqual(
            PipelineRequest(
                thread_id="t3",
                input_type="modification",
                prompt="add engraving",
                image_url="sample.png",
            ).input_type,
            "image_and_text",
        )
        self.assertEqual(
            PipelineRequest(thread_id="t4", input_type="image_only", image_url="sample.png").input_type,
            "image_only",
        )
        self.assertEqual(
            PipelineRequest(
                thread_id="t5",
                input_type="image_and_text",
                prompt="add gem",
                image_url="sample.png",
            ).input_type,
            "image_and_text",
        )

    def test_waiting_for_user_edit_is_valid_response_status(self):
        from src.llm_pipeline.core.schemas import PipelineResponse

        response = PipelineResponse(
            status="waiting_for_user_edit",
            optimized_image_urls=[],
            message="review edit",
            base_image_url="edited.png",
        )

        self.assertEqual(response.status, "waiting_for_user_edit")

    def test_start_requires_prompt_or_image(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        with self.assertRaises(ValueError):
            PipelineRequest(thread_id="t-start", action="start")

    def test_thread_id_is_required(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        with self.assertRaises(ValueError):
            PipelineRequest(thread_id="", prompt="ring idea")

    def test_request_customization_requires_non_empty_prompt(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        with self.assertRaises(ValueError):
            PipelineRequest(thread_id="t-custom", action="request_customization", customization_prompt="")


class FakeState:
    def __init__(self, values, next_nodes):
        self.values = values
        self.next = next_nodes


class FakeGraph:
    def __init__(self, state):
        self._state = state
        self.invocations = []
        self.updated_state = []

    def invoke(self, payload, config=None):
        self.invocations.append((payload, config))

    def update_state(self, config, values):
        self.updated_state.append((config, values))

    def get_state(self, config):
        return self._state


class FakeChatCompletions:
    def __init__(self, sink, content):
        self.sink = sink
        self.content = content

    def create(self, **kwargs):
        self.sink.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeEmbeddingsApi:
    def __init__(self, sink):
        self.sink = sink

    def create(self, **kwargs):
        self.sink.update(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in kwargs["input"]]
        )


class FakeOpenAIClient:
    def __init__(self, chat_sink=None, chat_content="", embed_sink=None):
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(chat_sink if chat_sink is not None else {}, chat_content)
        )
        self.embeddings = FakeEmbeddingsApi(embed_sink if embed_sink is not None else {})


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime dependencies are required for pipeline tests")
class PipelineRuntimeTests(unittest.TestCase):
    def test_vllm_multimodal_payload_uses_short_token_limit(self):
        vllm_client = importlib.import_module("src.llm_pipeline.core.vllm_client")
        config = importlib.import_module("src.llm_pipeline.core.config").config

        seen = {}
        original_chat_client = vllm_client._chat_client

        try:
            vllm_client._chat_client = lambda: FakeOpenAIClient(chat_sink=seen, chat_content='{"is_valid": true}')
            content = vllm_client.invoke_multimodal_json(
                prompt="judge the image",
                image_data_url="data:image/png;base64,abc123",
            )
        finally:
            vllm_client._chat_client = original_chat_client

        self.assertEqual(content, '{"is_valid": true}')
        self.assertEqual(seen["model"], config.VLLM_CHAT_MODEL)
        self.assertEqual(seen["max_tokens"], config.VLLM_VALIDATOR_MAX_TOKENS)
        self.assertEqual(seen["temperature"], 0.0)
        self.assertEqual(seen["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(seen["messages"][0]["content"][1]["type"], "image_url")
        self.assertEqual(
            seen["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,abc123",
        )

    def test_vllm_text_prompt_uses_prompt_token_limit(self):
        vllm_client = importlib.import_module("src.llm_pipeline.core.vllm_client")
        config = importlib.import_module("src.llm_pipeline.core.config").config

        seen = {}
        original_chat_client = vllm_client._chat_client

        try:
            vllm_client._chat_client = lambda: FakeOpenAIClient(chat_sink=seen, chat_content="ring, black background")
            content = vllm_client.invoke_text_prompt("rewrite this prompt")
        finally:
            vllm_client._chat_client = original_chat_client

        self.assertEqual(content, "ring, black background")
        self.assertEqual(seen["model"], config.VLLM_CHAT_MODEL)
        self.assertEqual(seen["max_tokens"], config.VLLM_PROMPT_MAX_TOKENS)
        self.assertEqual(seen["messages"][0]["content"], "rewrite this prompt")

    def test_vllm_embedding_function_returns_vectors(self):
        vllm_client = importlib.import_module("src.llm_pipeline.core.vllm_client")

        seen = {}
        embedder = vllm_client.VLLMEmbeddingFunction(model="embed-model", base_url="http://embed", api_key="token")
        original_client = embedder._client

        try:
            embedder._client = lambda: FakeOpenAIClient(embed_sink=seen)
            docs = embedder.embed_documents(["a", "b"])
            query = embedder.embed_query("c")
        finally:
            embedder._client = original_client

        self.assertEqual(seen["model"], "embed-model")
        self.assertEqual(seen["input"], ["c"])
        self.assertEqual(len(docs), 2)
        self.assertEqual(query, [0.1, 0.2, 0.3])

    def test_rag_uses_vllm_embedding_function(self):
        rag = importlib.import_module("src.llm_pipeline.nodes.rag")
        vllm_client = importlib.import_module("src.llm_pipeline.core.vllm_client")

        seen = {}

        class FakeChroma:
            def __init__(self, collection_name, embedding_function, persist_directory):
                seen["collection_name"] = collection_name
                seen["embedding_function"] = embedding_function
                seen["persist_directory"] = persist_directory

            def similarity_search(self, query, k=3):
                return []

        original_chroma = rag.Chroma
        original_exists = rag.os.path.exists

        try:
            rag.Chroma = FakeChroma
            rag.os.path.exists = lambda path: True
            rag.RingVectorRAG()
        finally:
            rag.Chroma = original_chroma
            rag.os.path.exists = original_exists

        self.assertEqual(seen["collection_name"], "ring_gemma_rules")
        self.assertIsInstance(seen["embedding_function"], vllm_client.VLLMEmbeddingFunction)

    def test_generation_system_error_stops_without_retry(self):
        agent = importlib.import_module("src.llm_pipeline.agent")

        self.assertEqual(
            agent.check_base_validation({"generation_result": "system_error", "retry_count": 0, "is_valid": False}),
            "end",
        )
        self.assertEqual(
            agent.check_edit_validation({"generation_result": "system_error", "retry_count": 0, "is_valid": False}),
            "end",
        )
        self.assertEqual(
            agent.check_rembg_validation({"generation_result": "system_error", "retry_count": 0, "is_valid": False}),
            "end",
        )

    def test_input_image_guardrail_routes_system_error_to_end(self):
        agent = importlib.import_module("src.llm_pipeline.agent")

        self.assertEqual(
            agent.check_input_image_processing({"guardrail_result": "system_error", "intent": "partial_modification"}),
            "end",
        )
        self.assertEqual(
            agent.check_input_image_processing({"guardrail_result": "repair_required", "intent": "multi_view_only"}),
            "edit_image",
        )
        self.assertEqual(
            agent.check_input_image_processing({"guardrail_result": "pass", "intent": "multi_view_only"}),
            "generate_multi_view",
        )

    def test_validate_base_image_checks_background_contrast(self):
        validator = importlib.import_module("src.llm_pipeline.nodes.validator")

        seen = {}
        original_call = validator._call_vision_judge

        try:
            validator._call_vision_judge = lambda image_url, prompt: (
                seen.update({"image_url": image_url, "prompt": prompt}) or {"is_valid": True, "reason": "ok"}
            )
            result = validator.validate_base_image(
                {
                    "base_ring_image_url": "base.png",
                    "user_prompt": "thin platinum ring with inside engraving",
                    "synthesized_prompt": "platinum ring, solid black background",
                    "retry_count": 0,
                }
            )
        finally:
            validator._call_vision_judge = original_call

        self.assertTrue(result["is_valid"])
        self.assertEqual(seen["image_url"], "base.png")
        self.assertIn("complementary or otherwise strongly contrasting", seen["prompt"])
        self.assertIn("white/platinum/silver", seen["prompt"])

    def test_validate_input_image_distinguishes_system_error_from_repair(self):
        validator = importlib.import_module("src.llm_pipeline.nodes.validator")

        original_call = validator._call_vision_judge

        try:
            validator._call_vision_judge = lambda image_url, prompt: {
                "is_valid": False,
                "reason": "vision unavailable",
                "result_type": "system_error",
            }
            result = validator.validate_input_image({"base_ring_image_url": "input.png"})

            self.assertEqual(result["guardrail_result"], "system_error")
            self.assertFalse(result["is_valid"])

            validator._call_vision_judge = lambda image_url, prompt: {
                "is_valid": False,
                "reason": "Change the background to solid pitch black",
                "result_type": "judged",
            }
            repaired = validator.validate_input_image(
                {
                    "base_ring_image_url": "input.png",
                    "user_prompt": "simple platinum ring",
                    "customization_prompt": "",
                }
            )
        finally:
            validator._call_vision_judge = original_call

        self.assertEqual(repaired["guardrail_result"], "repair_required")
        self.assertIn("solid pitch black", repaired["customization_prompt"])

    def test_synthesizer_enforces_strict_background_separation_prompt(self):
        synthesizer = importlib.import_module("src.llm_pipeline.nodes.synthesizer")

        prompt = synthesizer._enforce_background_contrast(
            "platinum wedding ring",
            "platinum wedding ring",
        )

        self.assertIn("single flat pure pitch-black studio background", prompt)
        self.assertIn("background color must strongly contrast with the ring material", prompt)
        self.assertIn("clearly visible empty inner hole", prompt)
        self.assertIn("single centered ring product photo", prompt)

    def test_edit_prompt_preserves_source_and_localizes_removal(self):
        synthesizer = importlib.import_module("src.llm_pipeline.nodes.synthesizer")

        prompt = synthesizer._compose_edit_prompt(
            {
                "customization_prompt": "remove the center diamond",
                "synthesized_prompt": "yellow gold solitaire ring, centered crop",
                "user_prompt": "remove the center diamond",
            },
            "Preserve pose and keep edits local.",
            "gemstone",
            "",
        )

        self.assertIn("Use the provided input ring image as the authoritative source.", prompt)
        self.assertIn("Keep the same ring identity, same composition, same camera angle, same crop", prompt)
        self.assertIn("Remove only the specifically requested detail", prompt)
        self.assertIn("Do not add replacement decorations", prompt)

    def test_validate_base_image_preserves_generation_system_error_message(self):
        validator = importlib.import_module("src.llm_pipeline.nodes.validator")

        result = validator.validate_base_image(
            {
                "base_ring_image_url": "",
                "generation_result": "system_error",
                "status_message": "ComfyUI request failed with HTTP 500: missing model",
                "retry_count": 0,
            }
        )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["generation_result"], "system_error")
        self.assertEqual(result["status_message"], "ComfyUI request failed with HTTP 500: missing model")

    def test_validator_resolves_plain_comfy_filename_to_input_view_url(self):
        validator = importlib.import_module("src.llm_pipeline.nodes.validator")

        seen = {}

        class FakeResponse:
            content = b"fake-image"

            def raise_for_status(self):
                return None

        original_get = validator.requests.get

        try:
            def fake_get(url, timeout=10):
                seen["url"] = url
                return FakeResponse()

            validator.requests.get = fake_get
            validator._encode_image_from_url("sample_input.png")
        finally:
            validator.requests.get = original_get

        self.assertIn("/view?filename=sample_input.png&type=input", seen["url"])

    def test_multi_view_only_edit_success_skips_second_wait(self):
        agent = importlib.import_module("src.llm_pipeline.agent")

        self.assertEqual(
            agent.check_edit_validation({"is_valid": True, "retry_count": 0, "intent": "multi_view_only"}),
            "generate_multi_view",
        )
        self.assertEqual(
            agent.check_edit_validation({"is_valid": True, "retry_count": 0, "intent": "partial_modification"}),
            "wait_for_edit_approval",
        )

    def test_start_request_copies_prompt_into_customization_for_image_and_text(self):
        schemas = importlib.import_module("src.llm_pipeline.core.schemas")
        pipelines = importlib.import_module("src.llm_pipeline.pipelines")

        fake_graph = FakeGraph(FakeState({"edited_ring_image_url": "edited.png"}, ("wait_for_edit_approval",)))
        original_graph = pipelines.app_graph
        pipelines.app_graph = fake_graph

        try:
            request = schemas.PipelineRequest(
                thread_id="thread-image-and-text",
                input_type="image_and_text",
                prompt="inside engraving forever",
                image_url="uploaded.png",
            )
            pipelines.process_generation_request(request)
        finally:
            pipelines.app_graph = original_graph

        initial_state = fake_graph.invocations[0][0]
        self.assertEqual(initial_state["customization_prompt"], "inside engraving forever")
        self.assertEqual(initial_state["input_type"], "image_and_text")

    def test_success_requires_non_empty_output_urls(self):
        schemas = importlib.import_module("src.llm_pipeline.core.schemas")
        pipelines = importlib.import_module("src.llm_pipeline.pipelines")

        fake_state = FakeState(
            {
                "is_valid": True,
                "final_output_urls": [],
                "status_message": "empty outputs",
                "base_ring_image_url": "base.png",
            },
            (),
        )
        fake_graph = FakeGraph(fake_state)
        original_graph = pipelines.app_graph
        pipelines.app_graph = fake_graph

        try:
            request = schemas.PipelineRequest(
                thread_id="thread-success-empty-outputs",
                input_type="text",
                prompt="slim platinum ring",
            )
            response = pipelines.process_generation_request(request)
        finally:
            pipelines.app_graph = original_graph

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.message, "empty outputs")

    def test_synthesizer_bridges_output_view_url_into_input_filename(self):
        synthesizer = importlib.import_module("src.llm_pipeline.nodes.synthesizer")

        seen = {}

        class FakeGetResponse:
            content = b"fake-image"

            def raise_for_status(self):
                return None

        class FakePostResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"name": "bridged_input.png"}

        original_get = synthesizer.requests.get
        original_post = synthesizer.requests.post

        try:
            def fake_get(url, timeout=20):
                seen["download_url"] = url
                return FakeGetResponse()

            def fake_post(url, files=None, timeout=20):
                seen["upload_url"] = url
                seen["upload_name"] = files["image"][0]
                return FakePostResponse()

            synthesizer.requests.get = fake_get
            synthesizer.requests.post = fake_post

            bridged = synthesizer._normalize_comfy_image_reference(
                "https://example-comfy/view?filename=ComfyUI_00003_.png&subfolder=&type=output"
            )
        finally:
            synthesizer.requests.get = original_get
            synthesizer.requests.post = original_post

        self.assertEqual(bridged, "bridged_input.png")
        self.assertIn("/upload/image", seen["upload_url"])
        self.assertEqual(seen["upload_name"], "ComfyUI_00003_.png")


@unittest.skipUnless(HAS_DB_DEPS, "db feeder dependencies are required for feeder tests")
class DbFeederContractTests(unittest.TestCase):
    def test_db_feeder_builds_vector_store_with_vllm_embeddings(self):
        db_feeder = importlib.import_module("src.llm_pipeline.scripts.db_feeder")
        vllm_client = importlib.import_module("src.llm_pipeline.core.vllm_client")

        fake_langchain_chroma = SimpleNamespace()
        seen = {}

        class FakeChroma:
            def __init__(self, collection_name, embedding_function, persist_directory):
                seen["collection_name"] = collection_name
                seen["embedding_function"] = embedding_function
                seen["persist_directory"] = persist_directory

        fake_langchain_chroma.Chroma = FakeChroma
        original_module = sys.modules.get("langchain_chroma")
        sys.modules["langchain_chroma"] = fake_langchain_chroma

        try:
            db_feeder._build_vector_store("vector-path")
        finally:
            if original_module is None:
                del sys.modules["langchain_chroma"]
            else:
                sys.modules["langchain_chroma"] = original_module

        self.assertEqual(seen["collection_name"], "ring_gemma_rules")
        self.assertEqual(seen["persist_directory"], "vector-path")
        self.assertIsInstance(seen["embedding_function"], vllm_client.VLLMEmbeddingFunction)

    def test_curated_rules_are_rich_and_stable(self):
        db_feeder = importlib.import_module("src.llm_pipeline.scripts.db_feeder")

        docs = db_feeder.CURATED_RULES
        ids = [doc["id"] for doc in docs]
        categories = {doc["category"] for doc in docs}

        self.assertGreaterEqual(len(docs), 20)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("Validation_and_Rembg", categories)
        self.assertIn("Ring_Material", categories)
        self.assertIn("background_subject_isolation_absolute", ids)
        self.assertIn("edit_preserve_pose_crop_background", ids)
        self.assertIn("edit_removal_restore_surface", ids)


if __name__ == "__main__":
    unittest.main()
