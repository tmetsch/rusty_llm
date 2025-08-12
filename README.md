
# Rust based AI LLM inference service

This repository contains all code to run a super simple AI LLM model - such as [Mistral 7b](https://mistral.ai/news/announcing-mistral-7b/); probably currently the 
best model to run locally - for inference; it includes simple RAG functionalities. Most importantly it exposes metrics 
about how long it took to create a response, as well as how long it took to generate the tokens.

![rusting llama being observed in mistral winds.](misc/inspecting_rusting_llama_in_mistral_wind.png)

## Warning

    This is for testing only; Use at your own risk! Main purpose is to learn hands-up on how this stuff works and to 
    intrument and characterize the behaviour of AI LLMs.

## Observability

The following key metrics are exposed through [Prometheus](https://prometheus.io/docs/practices/histograms/):

* *token_creation_duration* - Histogram for the time it took to generate the tokens.
* *inference_response_duration* - Histogram for the time it took to generate the full response (includes tokenization 
  and embedding additional context).
* *embedding_duration* - Histogram for the time it took to create a vector representation of the query and lookup 
  contextual information in the knowledge base.
* TODO: add more such as time it took to tokenize, read from KV store etc; also check if we can add tracing. 

Here is an example dashboard that capture the metrics described as well as some host metrics such as power, CPU
utilisation etc.:

![dashboard](misc/dashboard.png)

## Prerequisites

You will need to download a model and embedding model:

  * This [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main) 
    model seems to give reasonable good results. Otherwise, give the 
    [Phi-3.5-mini-instruct-Q4_K_S.gguf](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/tree/main) a try.
  * This [bge-base-en-v1.5.Q8_0.gguf](https://huggingface.co/ChristianAzinn/bge-base-en-v1.5-gguf/tree/main) 
    embedding model seem to work as well.

Best to put both files into a *model/* folder as *model.gguf* and *embed.gguf*.

## Configuration

This service can be configured through environment variables. The following variables are supported:

| Environment variable    | Description                                                           | Example/Default  |
|-------------------------|-----------------------------------------------------------------------|------------------|
| DATA_PATH               | Directory path from which to read text files into the knowledge base. | data             |
| EMBEDDING_MODEL         | Full path of the embedding model to use.                              | model/embed.gguf |
| HTTP_ADDRESS            | Bind address to use.                                                  | 127.0.0.1:8080   |
| HTTP_WORKERS            | Number of threads to run with the HTTP server.                        | 1                |
| INSTANCE_LABEL          | Label used to tag the prometheus metrics.                             | default          |
| MAIN_GPU                | Identifies which GPU we should use.                                   | 0                |
| MODEL_GPU_LAYERS        | Number of layers to offload to GPU.                                   | 0                |
| MODEL_MAX_TOKEN         | Maximum number of tokens to generate.                                 | 128              |
| MODEL_PATH              | Full path to the gguf file of the model.                              | model/model.gguf |
| MODEL_PROMPT_TEMPLATE   | A prompt template - should contain {context} and {query} elements.    | Mistral prompt   |
| MODEL_THREADS           | Number of threads we'll use for inference.                            | 6                |
| PROMETHEUS_HTTP_ADDRESS | Bind address to use for prometheus.                                   | 127.0.0.1:8081   |

Other environment variables such as RUST_LOG can also be used.

## Examples

The following [curl](https://curl.se/) commands show the format the service understands:

    $ curl -N -X POST http://localhost:8080/v1/chat/completions \
        -d '{"stream": true, "model": "rusty_llm", 
             "messages": [{"role": "user", "content": "Who was Albert Einstein?"}]}' \
        -H 'Content-Type: application/json'
    data: {"id":"foo","object":"chat.completion.chunk","created":1733007600,"model":"rusty_llm", "system_fingerprint": 
    "fp0", "choices":[{"index":0,"delta":{"content": " Albert"},"logprobs":null,"finish_reason":null}]}
    ...

You can also test if the RAG works by asking *Who was Thom Rhubarb?* - notice how easy it is to trick these word 
prediction machines - and see if respond with sth on screw-printers.

## Kubernetes based deployment

The [Dockerfile](Dockerfile) to build the image is best used on a machine with the same CPU as were it will be deployed 
as it uses *target-cpu=native* flag. Note that is also can optionally also include options to build with CLBlast for 
GPU support.

Use the following [example manifest](k8s_deployment.yaml) to deploy this application:

    kubectl apply -f k8s_deployment.yaml

***Note***: make sure to adapt the docker image & paths - the manifest above uses hostPaths!

## Changelog

  * 0.1.0 - initial release.
  * 0.2.0 - switch to [llama_cpp](https://github.com/edgenai/llama_cpp-rs) as [llm](https://github.com/rustformers/llm) stopped development.
  * 0.3.0 - replaced the way we store knowledge.
  * 0.4.0 - switch to [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) as it is under active development.
  * 0.5.0 - support to [OpenAI like API](https://platform.openai.com/docs/api-reference/introduction).
  * 0.6.0 - version fixes.
  * 0.7.0 - support for caching context etc.

## Further reading

Some of the following links can be useful:

  * https://docs.mistral.ai/guides/basic-RAG/
  * https://medium.com/@isalapiyarisi/lets-build-a-standalone-chatbot-with-phi-2-and-rust-48c0f714f915
