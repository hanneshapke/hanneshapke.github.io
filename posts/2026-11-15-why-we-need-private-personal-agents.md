---
layout: post.njk
title: Why We Need Private Personal Agents
description: An agent that does real work for you needs your email, your calendar, your documents and your finances. That context shouldn't live in someone else's cloud — and it no longer has to.
tags:   [ai agents,privacy,llm,open source,open weights,cedar]
---

*An agent that works for you needs to know everything about you. That's exactly why it should answer to you, and only to you.*

The chatbot era asked for our questions. The agent era asks for our lives.

A chatbot can get by with whatever you paste into the box. An agent that actually does things for you — plans a trip, reconciles your spending, drafts a reply to your landlord, keeps track of a medical appointment — needs standing access to your inbox, your calendar, your documents, your bank exports and your health data. It also needs to remember: who you are, what you prefer, what you asked last month. The more useful a personal agent becomes, the more complete the picture of you it has to hold.

Right now, the default place for that picture is someone else's data center. I think that's the wrong default, and I think it's worth saying why before it hardens into the only option.

## Your context is the product

Here's the uncomfortable part of the economics. The companies building consumer agents are spending extraordinary amounts on training and inference, and they need revenue paths that justify it. Subscriptions are one. Advertising is another: in January 2026, OpenAI [announced it would start testing ads in ChatGPT](https://fortune.com/2026/01/16/openai-to-begin-putting-ads-in-chatgpt/) for its free and Go tiers.

To be fair, OpenAI said conversations would stay private from advertisers and that ads wouldn't shape answers. I take that at face value. But the business model is what it is: once an assistant's provider earns money from attention, the most detailed record of a person's intentions ever assembled sits next to a team whose job is to monetize intent. Promises are policies, and policies change with leadership, with quarterly targets and with acquisitions. Architecture doesn't.

**A personal agent's memory is the most valuable dataset about you that will ever exist.** The question isn't whether today's provider is trustworthy. It's whether that dataset should exist on infrastructure where trust is the only safeguard.

## "Delete" doesn't mean what you think it means

The second problem isn't about anyone's intentions at all. It's about the law.

In May 2025, a federal magistrate judge in the New York Times copyright case ordered OpenAI to [preserve ChatGPT output logs](https://www.malwarebytes.com/blog/news/2025/06/openai-forced-to-preserve-chatgpt-chats) that would otherwise have been deleted, including chats users had deleted themselves and temporary sessions. OpenAI objected loudly, and Sam Altman responded by [calling for a form of "AI privilege"](https://venturebeat.com/ai/sam-altman-calls-for-ai-privilege-as-openai-clarifies-court-order-to-retain-temporary-and-deleted-chatgpt-sessions), comparable to what you'd have with a lawyer or doctor.

I agree with him that such a privilege should exist. But notice what the episode shows: users who pressed delete, and a company that wanted to honor that, were both overruled by a lawsuit neither user was party to. Your conversations with a hosted assistant are records held by a third party, and third-party records are reachable by subpoenas, discovery and preservation orders in ways that files on your own disk generally are not.

Now scale that up from chats to agents. It's no longer what you asked. It's your inbox, your calendar, your transaction history and a running memory of your habits, all sitting in one place, all discoverable.

## Agents raise the stakes on security, too

There's a third reason, and it's the one engineers should care about most.

Simon Willison has a name for the dangerous combination: the [lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/). An agent becomes exploitable when it has access to your private data, exposure to untrusted content, and a way to communicate externally. A useful personal agent has all three by design. It reads your email (private data), which includes email from strangers (untrusted content), and it can send messages and fetch URLs (external communication). One well-crafted message in your inbox can instruct the model to forward something it shouldn't.

Running the agent at home doesn't make prompt injection go away, and I want to be clear about that. What it changes is who controls the blast radius. When the agent is yours, you decide what it can reach: which tools are switched on, whether a task can see the internet, which actions need your explicit approval. When the agent lives in someone else's cloud, those decisions are made for you, for millions of users at once, by a product team optimizing for the median.

## What "private" should actually mean

"Private AI" is in danger of becoming a marketing term, so let me be specific about what I mean by a private personal agent:

1. **It runs on hardware you control.** Your laptop, a box under your desk, a home server. Your conversations, files and memory live there.
2. **You decide what leaves.** Using a hosted model can be perfectly reasonable — but it should be a choice you make, per setup, knowing that the provider then sees what the agent sends it. Nothing should leave by default.
3. **Every task runs in a sealed box.** The agent's tools execute in an isolated workspace that can't see your documents, your network or the internet unless you allow it.
4. **Risky actions need your consent.** Sending an email, deleting a file, downloading something: the agent stops and asks.
5. **Memory is inspectable.** You can see what it remembers about you, where each item came from, and delete any of it — and deletion means deletion.
6. **No telemetry, no account.** The software doesn't phone home, because there is no home other than yours.

None of these are exotic. They're the properties we'd demand from any other software that had this much access to our lives. We just haven't demanded them from agents yet.

## The hole in my argument

My recent posts have each had a section where I name the weakest part of the case. Here it is for this one.

**Local models are weaker.** The best frontier models still run in the labs' data centers. A model that fits on a laptop will not match them on every task, and for some work the difference matters.

But the gap has narrowed dramatically, and the open-weight models I wrote about [in my last post](/open-weights-arent-the-threat.html) are a large part of the reason. Most of what a personal agent does — reading, summarizing, scheduling, filling out a spreadsheet, writing a polite reply — doesn't need the absolute frontier. It needs a capable model, good tools and the right context. And the right context is precisely the part you shouldn't have to give away.

**Self-hosting is a hassle.** It used to require Docker, a GPU, a model server and an afternoon. If private agents only work for people who enjoy configuring things, they'll remain a niche, and the default will win by inertia. That's a real problem, and it's a product problem, not a law of nature. The answer is software that does the setup for you.

**Local doesn't mean safe.** As I said above, an agent on your own machine can still be tricked by a malicious email. Privacy and security are different properties, and a private agent still needs sandboxing, approvals and per-tool permissions. Running at home is necessary, not sufficient.

## What I've been building

I don't want to argue for this only in the abstract, so a disclosure: for the past months I've been building [Cedar](https://getcedar.io), an open-source personal assistant that tries to meet the bar above. It's a personal side project, Apache 2.0, and the code is on [GitHub](https://github.com/hanneshapke/cedar).

Cedar runs on a Mac, a Windows PC or a Linux box, with companion apps on iPhone and Android that pair with your own machine over Tailscale. Each conversation gets its own sandboxed workspace. It asks before sending email or deleting files, every tool has its own switch, and memory is a list you can read, edit and delete. It works with Ollama, LM Studio, llama.cpp or vLLM on your own hardware, and on a capable machine it can download llama.cpp and a fitting model for you. If you don't have the hardware, you can point it at a hosted provider instead — which is exactly the kind of informed choice I described above, made by you rather than for you. Its [privacy policy](https://github.com/hanneshapke/cedar/blob/main/PRIVACY.md) starts with one sentence: *cedar does not collect any data. Period.*

It's not the only way to build a private agent, and I'd be glad to see many more. The point isn't one project. The point is that this architecture is now practical.

## Final Thoughts

We are about to hand software the most intimate context we've ever given it. If we do that by default through a handful of cloud services, we'll have recreated the economics of the ad-funded web with a far more detailed profile at its center, and a far more capable actor operating on it.

It doesn't have to go that way. Open-weight models are good enough, consumer hardware is fast enough, and sandboxing is mature enough. A personal agent should run at home and answer to you. Let's build it like that before the alternative becomes the only one anyone remembers.

---

*Hannes Hapke is the Director of Dataiku's Open Source Office, 575 Lab. Cedar is a personal project and not affiliated with Dataiku.*
