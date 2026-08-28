---
layout: post.njk
title: Open Weights Aren't the Threat. Concentration Is.
description: Why the fight over Chinese open models is really a fight about who captures the value of AI — and why banning them is the worst available answer.
tags:   [open source,open weights,ai policy,llm,ai agents,economics,china]
---

*Why the fight over Chinese open models is really a fight about who captures the value of AI — and why banning them is the worst available answer.*

Here is the number that reframed this whole debate for me.

To justify their current valuations, the US frontier labs don't need to capture a larger share of the software market. They need to create a market that doesn't yet exist — one large enough to support nearly a trillion dollars in enterprise value at OpenAI and [Anthropic](https://www.anthropic.com/news/series-h) alone. Enterprise software worldwide is a few hundred billion dollars a year. Total compensation paid to American workers is on the order of [$16 trillion a year](https://fred.stlouisfed.org/series/COE) (Q1/2026).

You can't get from one to the other by selling seats. You get there by selling work.

That's not a conspiracy theory; it's the stated plan. When labs talk about agents that complete multi-day tasks, or about a "drop-in remote worker," they are describing a product whose addressable market is the wage bill. The frontier labs are not only competing with each other for a share of the revenue pie. They are competing with the labor market to determine how big that pie can get, and the pie has to get very large indeed.

I've spent most of my career building ML systems in production. I'm not writing this as a skeptic of the technology; I use these tools daily, and they're remarkable. But I think the financial structure being built on top of them is fragile in a specific way, and the thing most people are calling the threat is actually the release valve.

## The bet is more fragile than the valuations suggest

Roughly [45% of the S&P 500 market capitalization is now AI-linked](https://cryptonews.net/news/finance/32748613/), and AI-related names have driven the overwhelming majority of index gains this year. Most of the recent GDP growth has come from chip manufacturers and data center buildout. The capex boom is broad enough that even Caterpillar has caught a lift.

That concentration is the exposure. And the thing pressing on it isn't a competing US lab. It's a pricing floor collapsing from the other side of the Pacific.

Chinese labs — Moonshot AI with Kimi, [Z.ai](https://z.ai) with GLM-5.2, DeepSeek, and the rest — have been shipping open-weight models at or near frontier capability, under permissive licenses, at a fraction of the cost. Recent reporting puts Chinese open-weight models at [up to 46% of US enterprise token usage](https://finance.yahoo.com/technology/ai/articles/chinese-ai-models-now-capture-020440715.html). Not 46% of experiments. Production workloads, inside American companies, are running on weights trained in China.

Scott Galloway calls this AI dumping, and the label has stuck with me.

The financial logic of the worry is straightforward: if a good-enough model costs a hundredth as much and you can run it inside your own VPC (ignoring Kimi-3 and GLM-5.2 for a brief moment), the pricing power of the closed labs erodes. If pricing power erodes, the revenue path to those valuations closes. If that path closes while 45% of the index is priced for it, the correction isn't contained to a few private companies. It reaches pensions, payrolls, and the capex that has been carrying GDP growth.

I take that risk seriously. I just don't think the conclusion people are drawing from it follows.

## The solar analogy is the one worth arguing about

We have run this experiment before, and recently.

Early in my career, I had the chance to work for a European renewable energy company. During my time in the industry, Chinese manufacturers flooded the global market with solar panels below Western production costs and undercut the prices of European wind turbines. It was called dumping then, too. And it worked exactly as feared for the incumbents: Western solar manufacturing was gutted, European turbine manufacturers focused on risky deals like offshore, companies went bankrupt, and a strategic industry consolidated overseas.

It also collapsed the cost of decarbonization worldwide. Cheap panels did more for global emissions than any subsidy program of the era. But for everyone downstream of the panel — installers, utilities, households, and the climate — the dumping was a windfall.

<figure>
  <img src="/images/open-weights-arent-the-threat/solar-production-share.png" alt="Line chart: China's share of global solar production rises from 14% in 2006 to 75% by 2021, while Europe's falls from roughly 23% to 3%. EU tariffs go on around 2013, alongside the Q-Cells insolvency, and lapse around 2018.">
  <figcaption>Sources: Fraunhofer ISE, <em>Photovoltaics Report</em>; IEA, <em>Solar PV Global Supply Chains</em> (2022); ITIF (2020). The metric mixes cell- and module-production share: China's 2006–2013 figures are cell-production share (ITIF); its 2020–2021 figures and all of Europe's are module-production share (Fraunhofer ISE / IEA).</figcaption>
</figure>

The lesson I take from solar and wind isn't "cheap imports are good." It's that **protecting incumbents and building capacity are different policies, and the West chose the first one.** Tariffs on Chinese panels didn't rebuild American and European solar manufacturing. Sustained industrial investment eventually started to. We spent a decade defending a position instead of funding one.

The analogy isn't perfect. Solar panels are physical objects with shipping costs and tariff schedules. Model weights are files. You cannot put a tariff on a download, which means the protectionist option is even weaker here than it was there. But the structural shape holds: cheap supply is brutal for the incumbent layer and enormously valuable to everyone building on top of it.

The question is who you think you're governing for.

## What's actually on the table

When I first sketched these notes, I wrote that no one serious was proposing to ban open weights and that arguing against a ban was attacking a straw man. I was wrong about that, and the correction is the most important thing in this post.

Per [Axios](https://www.axios.com/2026/07/20/ai-us-china-open-source-kimi), parts of the administration are considering steps that would have been unthinkable to the open source world: adding Chinese AI labs to the Commerce Entity List; an executive order making US companies that host Chinese models liable for breaches; supply-chain regulations aimed at Chinese open-source models; and NSA advisories discouraging their use. Officials worried about stifling innovation had blocked these efforts. Personnel changes have since emboldened the hawks.

This matters because it inverts the usual framing. The pressure isn't only about national security. Here is White House AI adviser David Sacks, in that same reporting:

> "We are at a critical inflection point in AI policy. The leading closed labs, already a duopoly in AI model revenue, want the government to eliminate their open-source competition."

That is a sitting AI adviser describing a restriction campaign as regulatory capture. When the case for a ban is being made loudest by the firms whose valuations depend on there being no free alternative, the burden of proof shifts.

To be fair to the other side, and there is a real other side, the security concerns are not invented. Model weights carry the values and refusals of their training process, and a model trained under Chinese content rules brings those rules into your product. Provenance is genuinely hard to audit. Building critical infrastructure on a strategic rival's release cadence is a real dependency, even if the weights themselves sit on your disk.

But notice that most of those concerns argue for *inspection*, and open weights are the only artifacts you can actually inspect. You can red-team a downloaded model, run it air-gapped, fine-tune the behavior out, apply explainability tools like [Dataiku's Kiji Inspector](https://github.com/dataiku/kiji-inspector), and verify that nothing phones home. You can do none of that with an API. The security argument, followed honestly, is an argument for open weights and against dependency on any single provider, including the domestic ones.

A quick distinction worth keeping straight, since almost every article blurs it: most of these are **open weight**, not open source. DeepSeek ships under MIT, and Qwen under Apache 2.0, both of which are genuinely permissive. Llama shipped under a custom community license that the OSI does not consider open source at all. Training data and pipelines are rarely released by anyone. "Open" is a spectrum here, and pretending otherwise makes the policy conversation mushy.

## The uncomfortable question I have to answer

If you've followed me this far, there's a hole in my argument, and I'd rather name it than let you find it.

I opened by saying frontier labs are competing with the labor market. Then I argued for cheaper, more available models. But cost is the primary friction to automating a job. Making capable models nearly free doesn't slow labor substitution; it accelerates it. On its face, the thing I'm defending makes the thing I'm worried about worse.

I've sat with this, and I think the resolution is that **open weights don't change whether the work gets automated. They change who owns the automation.**

If the model layer stays closed and expensive, the surplus from every automated task routes through a duopoly and lands with its shareholders. If the model layer is commoditized, the value migrates to the application layer — to deployment, to domain expertise, to hospital systems, mid-sized manufacturers, universities, regional software firms, and individual practitioners. That's the classic move of commoditizing your complement, and it's why Meta open-sourced Llama in the first place. The work still changes. But the returns are distributed across thousands of organizations instead of accruing to three.

I'd rather face a disrupted labor market in which the tools are cheap and everyone can build with them than one in which the disruption is metered by a rent-collecting layer nobody can route around.

There's also a counterargument to my whole recession thesis that deserves airing: cheaper inference may *expand* total AI spending rather than shrink it; the Jevons paradox case Satya Nadella made after DeepSeek. If that holds, commoditized models grow the compute market rather than collapsing it, and NVIDIA is largely indifferent to whether the weights are open, because the tokens still run on someone's GPUs. I find this partly persuasive. It rescues the infrastructure layer. It does not rescue a valuation premised on selling intelligence at closed-lab margins.

## So what do we actually do

**Option one: restrict.** Entity listings, hosting liability, procurement bans. This buys a short-term floor under domestic pricing and doesn't stop the technology. Weights are files; capability diffuses; and bans tend to advertise the effectiveness of what they prohibit. Worse, the restriction would land on American companies rather than Chinese labs: the compliance costs fall on US firms that adopted these models because they were cheaper and about as good. And per Sacks, we'd be doing it substantially at the request of the incumbents it protects.

**Option two: build the alternative.** The reason American open models are losing isn't ideology; it's that nobody has a business model for giving away the output of a nine-figure training run. That's precisely the shape of the problem public funding exists to solve. We already have the skeleton: the National AI Research Resource, chronically underfunded relative to its mandate, and [NSF's PESOSE program](https://www.nsf.gov/tip/updates/nsf-investing-secure-open-source-ecosystems) at a scale that is a rounding error against a single frontier training run. If the US wants a domestic open ecosystem, it needs public compute at a state-of-the-art scale, available to universities, national labs, and small AI labs, with open release as a condition of access. The payoff isn't abstract: open weights running inside a hospital's own infrastructure is the only architecture under which much clinical AI is legally deployable at all. Europe, worth noting, has already legislated a version of this instinct; the EU AI Act carves out explicit exemptions for open-source models.

**Option three: deal with the economy we've actually built.** We opened this Pandora's box with ChatGPT and opened it further with coding agents, and I don't believe it closes. The outgoing Obama administration published a report in December 2016 that laid out almost exactly the dynamics we're now living through. It wasn't suppressed; it sits in the [White House archives](https://obamawhitehouse.archives.gov/sites/default/files/whitehouse_files/microsites/ostp/NSTC/preparing_for_the_future_of_ai.pdf) like every administration's work product does. It was simply shelved, which is worse in a way, because it means we had the analysis and declined to act on it for a decade.

Acting on it now means reskilling at a scale we've never attempted, publicly funded trade and apprenticeship programs (follow the German example), and — the omission that genuinely surprises me — AI as a school subject. Every elementary school student should be able to tell generated content from genuine content. Every high school student should understand what an open-weight model is, how to run one, and how to contribute to one. We teach media literacy. This is the same skill, on the technology that will define their working lives.

## Final Thoughts

Open source was never the threat. It's the oldest protection we have against information and technology monopolies, and it's the only part of this stack that a hospital, a school district, or a two-person startup can actually own.

The threat is the concentration of capability, of pricing power, and of the returns from automating work that people currently do. Right now, the most effective response to that concentration is a Chinese export. That should embarrass us into building our own, not into banning theirs.

---

*Hannes Hapke is the Director of Dataiku's Open Source Office, 575 Lab.*
