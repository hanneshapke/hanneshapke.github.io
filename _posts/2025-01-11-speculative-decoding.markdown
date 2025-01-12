---
layout: post
title:  Speculative Decoding with vLLM
description: Improving LLV inferences with speculative decoding
date:   2025-01-11 12:00:00 -0700
image:  '/images/speculative_decoding/speculative_decoding_comparison_header.jpg'
tags:   [machine learning,model deployment,vllm,speculative decoding,mlops,llm]
---

# Speculative Decoding with vLLM
When deploying large language models in production environments, latency optimization is crucial. This is particularly important for real-time applications like chatbots and conversational interfaces. While complex tasks often require larger LLMs (70 billion+ parameters), users still expect response times similar to smaller models. This challenge has led the machine learning community to continuously explore new ways to improve LLM latency.

One of the most promising techniques is speculative decoding, which is a technique that can be used to improve the performance of a language model by predicting multiple tokens at a time with a smaller model and use a larger model to validate the predictions.

## Problem

When serving large language models in production, latency optimization emerges as the biggest challenge that developers will face. This challenge becomes particularly acute in real-time scenarios, where users expect near-instantaneous interactions with chatbots, code completion tools, and other AI-powered interfaces.

At the heart of this challenge lies a fundamental characteristic of auto-regressive models: their sequential nature of text generation. Unlike many computational processes that can benefit from parallel processing, these models face an architectural constraint that proves to be their primary performance bottleneck. To generate any given token K, the model must first process and consider all preceding tokens, from 1 to K-1, in sequential order. This dependency chain, which provides context for the text generation, creates a processing pipeline that cannot be easily parallelized.

![Sequential token generation](/images/speculative_decoding/llm-token-generation.png)

Consider the process illustrated in Figure 1, where each token's generation depends on the complete history of previous tokens. This sequential dependency isn't merely a technical limitation—it's a fundamental aspect of how these models understand and generate human-like text.

The situation becomes even more challenging when we scale up to larger models, particularly those exceeding 3+ billion parameters. These massive models, while offering superior capabilities in terms of reasoning, understanding, and generation, exact a significant performance penalty. Each token prediction requires more computational resources, as the model must process its vast parameter space for every single token generation step. The result is a compounding effect: not only must we handle the sequential nature of token generation, but each step in that sequence now takes longer due to the model's size.

Yet, despite these performance challenges, larger models remain indispensable for many applications. They excel at complex tasks that smaller models struggle with, such as multi-step reasoning, code generation, and nuanced understanding of context. They also produce higher-quality text with fewer artifacts and better coherence. This creates a tension between the need for sophisticated model capabilities and the practical requirements of production deployment.

In this solution, we will use speculative decoding to improve the performance of LLM deployments. It will allow us to improve the model latency without changing the model architecture, training data, or the trained model itself.

## Solution

As we can see in Figure 1, most LLM deployments use tokens 1 to K-1 to generate token K. This process is sequential and slow and it process each token equally through the same model. One of key observations by Yaniv Leviathan et al. from Google [1] is that not every token needs this treatment. As they explained it in their paper, "some inference steps harder and others easier". They also made another observation that motivated their work: The processing isn't bound by computation, but rather by memory bandwidth. What's their solution?

Yaniv Leviathan et al. suggested to combine two models: a smaller model to predict the next tokens, a larger model to validate the prediction and, if needed, correct the prediction. The smaller model can also generate multiple tokens at a time, which is a great way to improve the overall latency.

Here is an example of how speculative decoding works:

> [START] Speculative decoding is ~~awesome~~ (corrected: a)

> [START] Speculative decoding is a technique that ~~could~~ (corrected: can)

> [START] Speculative decoding is a technique that can be used to ~~explain~~ (corrected: improve)

> [START] Speculative decoding is a technique that can be used to improve the performance of a language model.

We gain the inference speed increases by two aspects. First of all, we generate multiple tokens at once. We can request multiple tokens, since we have a second model to validate the predictions. We can afford it because the initial tokens predictions are fast and cheap. Secondly, the validation of the prediction is also fast, and we only need to correct the predictions for tokens where the smaller model made a mistake.

How can you use speculative decoding with your LLM? Most LLM deployment frameworks provide support of speculative decoding, in one form or another. For our core example, we are demonstrating speculative decoding with vLLM. vLLM is a frequently used framework for serving LLM models like Llama 3.2 3B. In our example, we use a smaller model to predict the next tokens and a larger model to validate the prediction and, if needed, correct the prediction. We use Meta's `opt-125m` model to predict the next tokens and the larger `opt-2.7b` model to validate the prediction and, if needed, correct the prediction. The sampling of the tokens, checking them with the larger model, and correcting them is done under the hood by the LLM serving framework, in our case vLLM.


```python

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-2.7b",
    tensor_parallel_size=1,
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```

When we compare the latency of speculative decoding with the latency of the same model without speculative decoding, we can see that speculative decoding is faster by roughly 35% as we can see in Figure 2.

![Latency comparison](/images/speculative_decoding/speculative_decoding_comparison.png)


## Trade-offs and Alternatives
There are significant trade-offs when using speculative decoding. The faster inferences don't come without downsides. In this section, we will discuss the trade-offs and present an alternative for hosting LLM models without speculative decoding.

### Larger Memory Footprint

The speculative decoding requires a larger memory footprint. The larger model needs to loaded into memory, together with the smaller model. This will require larger instances and GPUs, which translates to higher costs.
Also, speculative decoding can be difficult for fine-tuning models. The smaller model needs to be fine-tuned on the same dataset as the larger model. This is not always possible, since fine-tuning the larger model is more expensive than fine-tuning the smaller model (however, you might the performance boost from the smaller fine-tuned model already).

### Model Pairing

A larger model needs to be paired with a smaller model using the same tokenization. This is no problem for larger models to be paired with smaller models. However, this is not the case for smaller models.

The following table shows the possible combinations of base models and smaller models for speculative decoding.

| Base Model | Smaller Model for suggesting tokens |
|------------------------------|---------------------------|
| Llama 3.1 405b | Llama 3.1 70B |
| Llama 3.1 405b | Llama 3.2 3B |
| Llama 3.1 405b | Llama 3.2 1B |
| Llama 3.3 70b | Llama 3.2 3B |
| Llama 3.3 70b | Llama 3.2 1B |
| Llama 3.2 3B | Llama 3.1 70B |
| Llama 3.2 1B | ? |

### Problem specific Performance

The performance of speculative decoding also depends on the distribution of tokens. For example, if we want to generate English text for a chatbot, speculative decoding will be more effective than if we want to generate random hashes or bank transaction descriptions. in those cases, speculative decoding will actually be slower because the larger model needs to correct the predictions too often.

### Alternatives

A number of LLM serving frameworks support speculative decoding. Besides vLLM, SGLang also supports speculative decoding.
Here is a brief implementation of speculative decoding with SGLang.

```python
import sglang as sgl

prompts = [
    "Speculative decoding is",
]

    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 30}

    # Create an LLM.
    llm = sgl.Engine(
        model_path="facebook/opt-2.7b",
        speculative_draft_model_path="facebook/opt-125m",
        speculative_num_draft_tokens=5,
    )

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

## Demo

We have created a demo of speculative decoding with vLLM. You can find the code [in Colab](https://colab.research.google.com/drive/1CkI2Dl5WP2sEnspi8b4dV9J2lHl13sAl?usp=sharing).

## Conclusion

Speculative decoding is a promising technique to improve the performance of LLM deployments. in our demo example, we were able to improve the latency by 35%. However, it comes with a larger memory footprint and more deployment complexity.

## References

- [1] "Fast Inference from Transformers via Speculative Decoding", Yaniv Leviathan et al. [paper](https://arxiv.org/pdf/2211.17192), accessed January 11th, 2025.

## Suggested Readings
- "A Hitchhiker’s Guide to Speculative Decoding", [website](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/), accessed January 11th, 2025.
