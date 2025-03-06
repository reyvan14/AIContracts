# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import importlib

from comps import MegaServiceEndpoint, MicroService, ServiceOrchestrator, ServiceRoleType, ServiceType
from comps.cores.mega.utils import handle_message
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from comps.cores.proto.docarray import LLMParams, RerankerParms, RetrieverParms
from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.prompts import PromptTemplate

# 从 prompt.py 动态加载模板
def load_template(template_type):
    try:
        prompt_module = importlib.import_module('prompt')
        return prompt_module.pmts.get(template_type, None)
    except Exception as e:
        print(f"Error loading template {template_type}: {e}")
        return None


class ChatTemplate:
    @staticmethod
    def generate_rag_prompt(question, documents, template_type=None):
        """
        根据给定的 question 和检索到的 documents（上下文）生成最终给语言模型的提示信息（Prompt）。
        根据传入的模板类型动态选择合适的模板。
        """
        context_str = "\n".join(documents)

        # 加载指定模板
        if template_type:
            template = load_template(template_type)
            if template:
                return template.format(context=context_str, question=question)
        # 默认模板，如果没有传入有效的 template_type
        else:
            template = load_template('review')  # 默认使用 'review' 模板
            if template:
                return template.format(context=context_str, question=question)

        return ""  # 返回空字符串作为默认


# 以下环境变量用于指定各微服务的默认主机和端口，如果未在环境中指定，则使用默认值
MEGA_SERVICE_PORT = int(os.getenv("MEGA_SERVICE_PORT", 8888))
GUARDRAIL_SERVICE_HOST_IP = os.getenv("GUARDRAIL_SERVICE_HOST_IP", "0.0.0.0")
GUARDRAIL_SERVICE_PORT = int(os.getenv("GUARDRAIL_SERVICE_PORT", 80))
EMBEDDING_SERVER_HOST_IP = os.getenv("EMBEDDING_SERVER_HOST_IP", "0.0.0.0")
EMBEDDING_SERVER_PORT = int(os.getenv("EMBEDDING_SERVER_PORT", 80))
RETRIEVER_SERVICE_HOST_IP = os.getenv("RETRIEVER_SERVICE_HOST_IP", "0.0.0.0")
RETRIEVER_SERVICE_PORT = int(os.getenv("RETRIEVER_SERVICE_PORT", 7000))
RERANK_SERVER_HOST_IP = os.getenv("RERANK_SERVER_HOST_IP", "0.0.0.0")
RERANK_SERVER_PORT = int(os.getenv("RERANK_SERVER_PORT", 80))
LLM_SERVER_HOST_IP = os.getenv("LLM_SERVER_HOST_IP", "0.0.0.0")
LLM_SERVER_PORT = int(os.getenv("LLM_SERVER_PORT", 80))
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")


def align_inputs(self, inputs, cur_node, runtime_graph, llm_parameters_dict, **kwargs):
    """
    根据当前节点类型，将输入数据 align（对齐）为下游服务能够理解的格式。
    例如：embedding 服务需要 "inputs" 字段，llm 服务需要转换成 OpenAI /v1/chat/completions 兼容的格式等。
    """
    # 如果是 EMBEDDING 服务，需要将 text 替换为 inputs
    if self.services[cur_node].service_type == ServiceType.EMBEDDING:
        inputs["inputs"] = inputs["text"]
        del inputs["text"]
    elif self.services[cur_node].service_type == ServiceType.RETRIEVER:
        # 检索服务需要附带 retriever 参数（如 top_k、阈值等）
        retriever_parameters = kwargs.get("retriever_parameters", None)
        if retriever_parameters:
            inputs.update(retriever_parameters.dict())
    elif self.services[cur_node].service_type == ServiceType.LLM:
        # 统一转换为与 OpenAI /v1/chat/completions 兼容的输入格式
        next_inputs = {}
        next_inputs["model"] = LLM_MODEL
        next_inputs["messages"] = [{"role": "user", "content": inputs["inputs"]}]
        next_inputs["max_tokens"] = llm_parameters_dict["max_tokens"]
        next_inputs["top_p"] = llm_parameters_dict["top_p"]
        next_inputs["stream"] = inputs["stream"]
        next_inputs["frequency_penalty"] = inputs["frequency_penalty"]
        # next_inputs["presence_penalty"] = inputs["presence_penalty"]
        # next_inputs["repetition_penalty"] = inputs["repetition_penalty"]
        next_inputs["temperature"] = inputs["temperature"]
        inputs = next_inputs
    return inputs


def align_outputs(self, data, cur_node, inputs, runtime_graph, llm_parameters_dict, **kwargs):
    """
    根据服务类型，对返回结果做二次处理，输出给下游服务或返回给调用方。
    例如：retriever 服务输出多个文档，若下游是 rerank，则需将检索结果格式化给 rerank；若下游直接是 llm，则根据 prompt 模板拼接生成 prompt。
    """
    next_data = {}
    # 处理 EMBEDDING 服务的输出
    if self.services[cur_node].service_type == ServiceType.EMBEDDING:
        # 对 embedding 服务返回的结果（通常是一个向量列表）进行包装
        assert isinstance(data, list)
        next_data = {"text": inputs["inputs"], "embedding": data[0]}

    # 处理 RETRIEVER 服务的输出
    elif self.services[cur_node].service_type == ServiceType.RETRIEVER:
        # 获取检索到的文档
        docs = [doc["text"] for doc in data["retrieved_docs"]]

        # 判断下游是否有 rerank
        with_rerank = runtime_graph.downstream(cur_node)[0].startswith("rerank")
        if with_rerank and docs:
            # 如果有文档并且确实有 rerank 节点，就将检索结果打包给 rerank
            next_data["query"] = data["initial_query"]
            next_data["texts"] = [doc["text"] for doc in data["retrieved_docs"]]
        else:
            # 如果没有文档或不需要 rerank，则直接流向 llm
            if not docs and with_rerank:
                # 如果没有检索到结果，但流程图却包含 rerank，就删除 rerank，直接从 retriever 流向 llm
                for ds in reversed(runtime_graph.downstream(cur_node)):
                    for nds in runtime_graph.downstream(ds):
                        runtime_graph.add_edge(cur_node, nds)
                    runtime_graph.delete_node_if_exists(ds)

            # 处理用户自定义模板；如果没提供则默认使用 ChatTemplate 生成 RAG 提示
            prompt = data["initial_query"]
            chat_template = llm_parameters_dict["chat_template"]
            if chat_template:
                prompt_template = PromptTemplate.from_template(chat_template)
                input_variables = prompt_template.input_variables
                if sorted(input_variables) == ["context", "question"]:
                    prompt = prompt_template.format(question=data["initial_query"], context="\n".join(docs))
                elif input_variables == ["question"]:
                    prompt = prompt_template.format(question=data["initial_query"])
                else:
                    print(f"{prompt_template} not used, we only support 2 input variables ['question', 'context']")
                    prompt = ChatTemplate.generate_rag_prompt(data["initial_query"], docs)
            else:
                prompt = ChatTemplate.generate_rag_prompt(data["initial_query"], docs)

            next_data["inputs"] = prompt

    # 处理 RERANK 服务的输出
    elif self.services[cur_node].service_type == ServiceType.RERANK:
        # 从输入中拿到之前检索的 docs
        reranker_parameters = kwargs.get("reranker_parameters", None)
        top_n = reranker_parameters.top_n if reranker_parameters else 1
        docs = inputs["texts"]

        # data 里存储了重排后的结果（每个结果带 index）
        reranked_docs = []
        for best_response in data[:top_n]:
            reranked_docs.append(docs[best_response["index"]])

        # 处理模板，与上面相似
        prompt = inputs["query"]
        chat_template = llm_parameters_dict["chat_template"]
        if chat_template:
            prompt_template = PromptTemplate.from_template(chat_template)
            input_variables = prompt_template.input_variables
            if sorted(input_variables) == ["context", "question"]:
                prompt = prompt_template.format(question=prompt, context="\n".join(reranked_docs))
            elif input_variables == ["question"]:
                prompt = prompt_template.format(question=prompt)
            else:
                print(f"{prompt_template} not used, we only support 2 input variables ['question', 'context']")
                prompt = ChatTemplate.generate_rag_prompt(prompt, reranked_docs)
        else:
            prompt = ChatTemplate.generate_rag_prompt(prompt, reranked_docs)

        next_data["inputs"] = prompt

    # 处理 LLM（大模型）非流式输出场景
    elif self.services[cur_node].service_type == ServiceType.LLM and not llm_parameters_dict["stream"]:
        next_data["text"] = data["choices"][0]["message"]["content"]

    # 其他情况直接将 data 返回
    else:
        next_data = data

    return next_data


def align_generator(self, gen, **kwargs):
    """
    将后台返回的流式数据（可能是像 TGI/vLLM 等流式输出）转换为兼容 OpenAI 的 SSE 格式（Server-Sent Events）。
    从中提取模型生成的部分，并以 data: {...}\n\n 形式逐行返回给前端。
    """
    for line in gen:
        line = line.decode("utf-8")
        start = line.find("{")
        end = line.rfind("}") + 1

        json_str = line[start:end]
        try:
            # 有时会遇到空的 chunk 或不完整的 JSON，可以做异常捕获
            json_data = json.loads(json_str)
            if (
                json_data["choices"][0]["finish_reason"] != "eos_token"
                and "content" in json_data["choices"][0]["delta"]
            ):
                # 将生成的内容包装为 SSE
                yield f"data: {repr(json_data['choices'][0]['delta']['content'].encode('utf-8'))}\n\n"
        except Exception as e:
            # 避免因为部分解析失败导致崩溃，直接把原始 json_str 返回
            yield f"data: {repr(json_str.encode('utf-8'))}\n\n"
    yield "data: [DONE]\n\n"


class ChatQnAService:
    """
    ChatQnAService 封装了一个基于 ServiceOrchestrator 的“聊天问答”服务：
    1. 可以添加 embedding、retriever、rerank、llm 等远程服务
    2. 可以处理客户端请求，将其打包后按照编排流程调用微服务
    3. 最后将结果以标准 OpenAI chat completion 的形式返回
    """

    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        # 将我们自定义的 align_* 方法绑定到 orchestrator 中
        ServiceOrchestrator.align_inputs = align_inputs
        ServiceOrchestrator.align_outputs = align_outputs
        ServiceOrchestrator.align_generator = align_generator
        self.megaservice = ServiceOrchestrator()
        self.endpoint = str(MegaServiceEndpoint.CHAT_QNA)

    def add_remote_service(self):
        """
        默认添加包含 embedding -> retriever -> rerank -> llm 的四节点服务编排流程
        """
        embedding = MicroService(
            name="embedding",
            host=EMBEDDING_SERVER_HOST_IP,
            port=EMBEDDING_SERVER_PORT,
            endpoint="/embed",
            use_remote_service=True,
            service_type=ServiceType.EMBEDDING,
        )

        retriever = MicroService(
            name="retriever",
            host=RETRIEVER_SERVICE_HOST_IP,
            port=RETRIEVER_SERVICE_PORT,
            endpoint="/v1/retrieval",
            use_remote_service=True,
            service_type=ServiceType.RETRIEVER,
        )

        rerank = MicroService(
            name="rerank",
            host=RERANK_SERVER_HOST_IP,
            port=RERANK_SERVER_PORT,
            endpoint="/rerank",
            use_remote_service=True,
            service_type=ServiceType.RERANK,
        )

        llm = MicroService(
            name="llm",
            host=LLM_SERVER_HOST_IP,
            port=LLM_SERVER_PORT,
            endpoint="/v1/chat/completions",
            use_remote_service=True,
            service_type=ServiceType.LLM,
        )
        # 将所有微服务加入 orchestrator
        self.megaservice.add(embedding).add(retriever).add(rerank).add(llm)
        # 指定服务调用顺序
        self.megaservice.flow_to(embedding, retriever)
        self.megaservice.flow_to(retriever, rerank)
        self.megaservice.flow_to(rerank, llm)

    def add_remote_service_without_rerank(self):
        """
        添加 embedding -> retriever -> llm 的三节点流程（不包含 rerank）
        """
        embedding = MicroService(
            name="embedding",
            host=EMBEDDING_SERVER_HOST_IP,
            port=EMBEDDING_SERVER_PORT,
            endpoint="/embed",
            use_remote_service=True,
            service_type=ServiceType.EMBEDDING,
        )

        retriever = MicroService(
            name="retriever",
            host=RETRIEVER_SERVICE_HOST_IP,
            port=RETRIEVER_SERVICE_PORT,
            endpoint="/v1/retrieval",
            use_remote_service=True,
            service_type=ServiceType.RETRIEVER,
        )

        llm = MicroService(
            name="llm",
            host=LLM_SERVER_HOST_IP,
            port=LLM_SERVER_PORT,
            endpoint="/v1/chat/completions",
            use_remote_service=True,
            service_type=ServiceType.LLM,
        )
        self.megaservice.add(embedding).add(retriever).add(llm)
        self.megaservice.flow_to(embedding, retriever)
        self.megaservice.flow_to(retriever, llm)

    def add_remote_service_with_guardrails(self):
        """
        在服务编排中加入 guardrail_in 作为输入拦截服务，用于敏感内容过滤或安全控制。
        同时，后续依旧包含 embedding -> retriever -> rerank -> llm 的链路。
        """
        guardrail_in = MicroService(
            name="guardrail_in",
            host=GUARDRAIL_SERVICE_HOST_IP,
            port=GUARDRAIL_SERVICE_PORT,
            endpoint="/v1/guardrails",
            use_remote_service=True,
            service_type=ServiceType.GUARDRAIL,
        )
        embedding = MicroService(
            name="embedding",
            host=EMBEDDING_SERVER_HOST_IP,
            port=EMBEDDING_SERVER_PORT,
            endpoint="/embed",
            use_remote_service=True,
            service_type=ServiceType.EMBEDDING,
        )
        retriever = MicroService(
            name="retriever",
            host=RETRIEVER_SERVICE_HOST_IP,
            port=RETRIEVER_SERVICE_PORT,
            endpoint="/v1/retrieval",
            use_remote_service=True,
            service_type=ServiceType.RETRIEVER,
        )
        rerank = MicroService(
            name="rerank",
            host=RERANK_SERVER_HOST_IP,
            port=RERANK_SERVER_PORT,
            endpoint="/rerank",
            use_remote_service=True,
            service_type=ServiceType.RERANK,
        )
        llm = MicroService(
            name="llm",
            host=LLM_SERVER_HOST_IP,
            port=LLM_SERVER_PORT,
            endpoint="/v1/chat/completions",
            use_remote_service=True,
            service_type=ServiceType.LLM,
        )

        # 将各服务加入 orchestrator
        self.megaservice.add(guardrail_in).add(embedding).add(retriever).add(rerank).add(llm)
        # 指定服务调用顺序
        self.megaservice.flow_to(guardrail_in, embedding)
        self.megaservice.flow_to(embedding, retriever)
        self.megaservice.flow_to(retriever, rerank)
        self.megaservice.flow_to(rerank, llm)

    async def handle_request(self, request: Request):
        """
        处理 POST 请求：
        1. 解析请求为 ChatCompletionRequest
        2. 将用户的 messages 拼接成 prompt
        3. 将参数包装为 LLMParams、RetrieverParms、RerankerParms
        4. 交给 orchestrator 来执行
        5. 将结果格式化成 ChatCompletionResponse 返回
        """
        data = await request.json()

        # 获取 template_type 参数
        template_type = data.get("template_type", None)

        stream_opt = data.get("stream", True)
        chat_request = ChatCompletionRequest.parse_obj(data)

        # 获取 prompt 内容，根据 template_type 动态加载
        prompt = handle_message(chat_request.messages)

        # 从 prompt.py 加载对应模板内容
        if template_type:
            prompt_template = ChatTemplate.generate_rag_prompt(chat_request.messages[0]["content"], [],
                                                               template_type=template_type)
        else:
            prompt_template = ChatTemplate.generate_rag_prompt(chat_request.messages[0]["content"], [])

        # LLM 参数，包含对话模板、流式输出选项、温度、惩罚项等
        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            stream=stream_opt,
            chat_template=chat_request.chat_template if chat_request.chat_template else None,
        )

        retriever_parameters = RetrieverParms(
            search_type=chat_request.search_type if chat_request.search_type else "similarity",
            k=chat_request.k if chat_request.k else 4,
            distance_threshold=chat_request.distance_threshold if chat_request.distance_threshold else None,
            fetch_k=chat_request.fetch_k if chat_request.fetch_k else 20,
            lambda_mult=chat_request.lambda_mult if chat_request.lambda_mult else 0.5,
            score_threshold=chat_request.score_threshold if chat_request.score_threshold else 0.2,
        )

        reranker_parameters = RerankerParms(
            top_n=chat_request.top_n if chat_request.top_n else 1,
        )

        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"text": prompt_template},
            llm_parameters=parameters,
            retriever_parameters=retriever_parameters,
            reranker_parameters=reranker_parameters,
        )

        # 如果在编排中出现流式响应（StreamingResponse），直接返回
        for node, response in result_dict.items():
            if isinstance(response, StreamingResponse):
                return response

        # 否则获取最终叶子节点的返回结果，包装为 OpenAI ChatCompletionResponse 格式
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="chatqna", choices=choices, usage=usage)

    def start(self):
        """
        启动当前的服务（ChatQnAService），对外提供 HTTP 接口。
        """
        self.service = MicroService(
            self.__class__.__name__,
            service_role=ServiceRoleType.MEGASERVICE,
            host=self.host,
            port=self.port,
            endpoint=self.endpoint,
            input_datatype=ChatCompletionRequest,
            output_datatype=ChatCompletionResponse,
        )

        # 将 handle_request 方法绑定到指定 endpoint
        self.service.add_route(self.endpoint, self.handle_request, methods=["POST"])

        # 启动服务
        self.service.start()


if __name__ == "__main__":
    # 解析命令行参数，根据是否有 --without-rerank 或 --with-guardrails 来决定添加的微服务组合
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-rerank", action="store_true")
    parser.add_argument("--with-guardrails", action="store_true")

    args = parser.parse_args()

    chatqna = ChatQnAService(port=MEGA_SERVICE_PORT)
    if args.without_rerank:
        chatqna.add_remote_service_without_rerank()
    elif args.with_guardrails:
        chatqna.add_remote_service_with_guardrails()
    else:
        chatqna.add_remote_service()

    chatqna.start()
