---
layout: post.njk
title: Kiji Privacy Proxy™ - Protecting Your Data in the Age of Generative AI
description: Introducing Kiji Privacy Proxy, an open-source gateway that automatically detects and redacts PII before it reaches external LLM APIs, letting enterprises use generative AI without exposing sensitive data.
tags:   [machine learning,privacy,pii,llm,generative ai,dataiku,open source,kiji]
---

*Originally published on the [Dataiku Blog](https://www.dataiku.com/stories/blog/kiji-privacy-proxy).*

Every time you type a prompt into ChatGPT, Claude, or any other LLM-powered service, you're sending data to an external server. For casual questions, that's fine. But in enterprise settings, those prompts often contain customer names, email addresses, social security numbers, medical records, financial details, and internal business data that should never leave your environment.

This isn't a hypothetical risk. Regulations like GDPR, HIPAA, and CCPA impose real penalties on organizations that fail to protect personal data, and sending PII to a third-party API without proper safeguards can constitute a violation. A 2026 Dataiku/Harris Poll study of 600 CIOs found that 85% have seen AI projects delayed or blocked entirely due to gaps in traceability or explainability, and privacy concerns are a major part of that picture.

The challenge is clear: Enterprises want the productivity gains of generative AI, but they can't afford to expose sensitive data in the process. Until now, the main options have been to either avoid using external AI services altogether (losing the benefits), build expensive custom infrastructure, or simply accept the risk and hope for the best.

None of those options is good enough.

## Why Kiji Privacy Proxy™

Operating as a transparent gateway between your local applications and external AI APIs, Kiji Privacy Proxy™ ensures you don't have to compromise your workflow or abandon powerful AI tools. By sitting directly within your network, Kiji automatically identifies and redacts personally identifiable information (PII) before any data is transmitted, allowing you to leverage generative AI without having to trust third-party servers with your sensitive information.

Here's how it works: Your app sends a request to the Kiji Privacy Proxy, and it forwards it to services like OpenAI or Anthropic. Alternatively, Kiji can intercept requests as well, run them through an ML-powered PII detection model, and replace any sensitive data, emails, phone numbers, credit card numbers, SSNs, IP addresses, and 16+ other PII types with realistic dummy values. The masked request goes out to the API. When the response comes back, Kiji restores the original values so your application works exactly as expected.

The result: The AI model never sees your real data, but your application behaves as though nothing changed.

What makes Kiji particularly practical is how little friction it introduces. On macOS, it runs as a native desktop app with automatic proxy configuration. We also provide a Chrome extension that routes web requests through Kiji without any environment variables or code changes. On Linux, it runs as a standalone server. In both cases, latency stays under 100 milliseconds for most requests, and all PII detection happens locally with no external API calls.

Kiji is open source under the Apache 2.0 license, and both the trained model and its training dataset are published on HuggingFace ([DataikuNLP/kiji-pii-model-onnx](https://huggingface.co/DataikuNLP/kiji-pii-model-onnx) and [DataikuNLP/kiji-pii-training-data](https://huggingface.co/DataikuNLP/kiji-pii-training-data)), so you can inspect, reproduce, and extend everything.

The Kiji Privacy Proxy is powered by a base model (developed by Dataiku's 575 Lab) that attained a 94% F1 score on the industry benchmark dataset. This result is highly competitive when compared to similar models in the field.

## A Collaboration Between Forward-Thinking ML Companies

Kiji Privacy Proxy doesn't exist in a vacuum. It's part of a broader vision where specialized companies across the ML ecosystem each contribute what they do best, and the result is greater than the sum of its parts. Built by Dataiku's 575 Lab, Kiji draws on and connects with the work of several outstanding partners.

**Dataiku** — The company that created Kiji, brings over a decade of enterprise AI experience, and recently launched the 575 Lab as its Open Source Office, is dedicated to building deployable tools for AI transparency, privacy, and governance. Kiji is one of the Lab's first releases, alongside agent explainability tools. As a member of the Linux Foundation and the Agentic AI Foundation, Dataiku, the Platform for AI Success, is committed to building these capabilities in the open.

**Outerbounds** — The company behind Metaflow, the open-source ML infrastructure stack originally built at Netflix, provides state-of-the-art infrastructure that makes complex ML workflows manageable. For teams that want to integrate Kiji's PII detection into larger ML pipelines, train custom models, orchestrate data flows, and manage deployment at scale, Outerbounds' infrastructure-as-code approach is a natural complement.

**HumanSignal** — The creators of Label Studio, the world's most popular open-source data labeling tool used by over 350,000 researchers, plays a critical role in the data quality side of the equation. Kiji's ML model is only as good as its training data, and for organizations that need to customize PII detection for their specific domain (think medical record formats, industry-specific identifiers, or non-English PII patterns), Label Studio provides the labeling infrastructure to build and refine those custom datasets.

**Doubleword** — The inference provider for high-volume workloads, founded by researchers from Oxford University who pioneered techniques in model optimization, completes the picture on the deployment side. Doubleword's inference platform offers open-source model inference at a fraction of the cost of other providers, making it well-suited for high-volume workloads such as data and document processing, as well as async agents. In this case, Doubleword models were used to generate large volumes of synthetic data at a cost of only $50 — just five percent of what comparable models from closed-source providers would have cost.

## Make Your Domain-Specific Kiji Privacy Proxy

One of the most powerful aspects of Kiji is its design for customization. The default model handles common PII types well, but every industry has unique data patterns that a generic model won't catch, such as pharmaceutical compound identifiers, internal project codes, proprietary customer reference numbers, and jurisdiction-specific ID formats.

Kiji's architecture makes it straightforward to build your own domain-specific privacy proxy. The training data and model are fully open on HuggingFace. With Doubleword's batch inference platform, you can create your own large synthetic data. You can use Label Studio (by HumanSignal) to annotate the domain-specific, synthetic PII examples. You can orchestrate the training pipeline with Metaflow (by Outerbounds) on whatever compute you need.

This is what collaboration across the ML ecosystem looks like in practice: not a single monolithic product, but a set of interoperable, open tools built by companies that deeply understand their piece of the puzzle. Together, they give enterprises the building blocks to protect their data without sacrificing the transformative potential of generative AI.

## Star the Repo on GitHub and Try It Out

Read the full post on the [Dataiku blog](https://www.dataiku.com/stories/blog/kiji-privacy-proxy).
