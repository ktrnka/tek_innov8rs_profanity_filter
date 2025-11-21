
## Problem Space Analysis

### Core Challenges

**1. Curse Words \& Profanity**

- Traditional profanity filters struggle with **obfuscation techniques**: leetspeak (h3ll, sh1t), character substitution (f*ck, a\$\$), spacing (f u c k), phonetic variations
- **Context matters**: "That's sick!" (positive) vs "You're sick" (negative)
- **False positives**: Legitimate words containing profane substrings (Scunthorpe problem, "assist", "mishit")

**2. Sexual Content**

- Explicit sexual terms and innuendo
- Often disguised through creative spellings or acronyms
- Cultural variations in what's considered sexual content

**3. Hate Speech \& Harassment**

- **Targeted attacks**: Against specific individuals, groups, or protected classes
- **Dog whistles**: Coded language that appears innocent but carries hateful meaning
- **Implicit vs explicit**: Direct slurs vs subtle dehumanization
- Research shows models often rely too heavily on keyword detection and miss implicit abuse[^1]

**4. Multilingual Challenges**

- **Code-mixing**: Users switching between languages mid-conversation
- **Transliteration**: Writing non-Latin scripts using Latin characters
- **Cultural context**: Same word may be offensive in one culture but not another
- **Resource imbalance**: Most datasets are English-heavy; low-resource languages underserved[^2][^3]

**5. Username/Guild Name Specific Issues**

- **No linguistic context**: Unlike chat, usernames lack surrounding text for context
- **Creative obfuscation**: "ifyouseekamy" (say it aloud), "tackhilla"[^4]
- **Legitimate names that trigger filters**: "Ho", "Kuntz", "Phuoc", "Gay" are real surnames[^5]
- **Mixed results in practice**: ADL study found major games (League of Legends, PUBG, Fortnite, Call of Duty, Overwatch 2) had inconsistent username moderation[^6]
- **Personalized license plate problem**: Users create intentionally ambiguous combinations

**6. Gaming-Specific Context**

- **Extremely short messages**: In-game chat is terse, making context detection harder[^7]
- **Game-specific slang**: Terms that are toxic in-game but innocuous elsewhere
- **Competitive trash talk**: Blurred line between acceptable banter and harassment
- **Real-time requirements**: Need fast detection for live moderation

***

## Available Labeled Datasets

### General Toxicity \& Hate Speech

| Dataset | Size | Source | Labels | Access |
| :-- | :-- | :-- | :-- | :-- |
| **HateXplain**[^8] | 20,148 posts | Twitter, Gab | Hate/Offensive/Normal + target community + rationales | [HuggingFace](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) |
| **Measuring Hate Speech**[^9] | ~50,000 annotations on 10,000+ comments | YouTube, Reddit, Twitter | Continuous hate scale + 10 survey items + demographics | [HuggingFace](https://hatespeechdata.com) |
| **ToxicChat**[^10] | 10,000 entries | Real user-AI interactions | Toxic/Non-toxic in conversational AI context | Public |
| **THOS**[^11] | 8,300 tweets | Twitter | Fine-grained targets of hate/offensive speech | Available |
| **HateCOT**[^12] | 52,000+ samples | Diverse sources | Offensive labels + GPT-generated explanations | Available |
| **Toxic Comment Classification**[^13] | Large-scale | Kaggle | Multi-label (toxic, severe_toxic, obscene, threat, insult, identity_hate) | [Kaggle](https://www.kaggle.com/) |
| **Hate Speech \& Offensive Language**[^14] | Twitter data | Twitter | Hate speech / Offensive / Neither | Kaggle |
| **Surge AI Toxicity**[^15] | 1,000 (500 toxic, 500 non-toxic) | Multiple social media | Binary toxic/non-toxic | [GitHub](https://github.com/surge-ai/toxicity) |

### Gaming-Specific Datasets

| Dataset | Size | Source | Labels | Access |
| :-- | :-- | :-- | :-- | :-- |
| **CONDA**[^16][^17] | 45K utterances from 12K conversations | Dota 2 match chats | Utterance \& token-level toxicity | Available (academic) |
| **GameTox**[^18] | 53,000 utterances | World of Tanks | Intent classification + slot filling | Available (NAACL 2025) |
| **GOSU.ai Dota 2**[^19] | 147,842 messages | Dota 2 | Clean (0) / Mild toxicity (1) / Strong toxicity (2) | Public |
| **WoW/LoL Forum Dataset**[^20] | Unknown size | World of Warcraft \& League of Legends forums | Cyberbullying/Abusive vs Normal | Available (83% F1-score achieved) |
| **Rainbow Six Siege, For Honor**[^21][^22] | Multiple game datasets | In-game chats | Toxic/Non-toxic with game context | Used in ToxBuster research |

### Multilingual Datasets

| Dataset | Languages | Size | Notes |
| :-- | :-- | :-- | :-- |
| **ADIMA**[^23] | 10 Indic languages | 11,775 audio samples (65 hours, 6,446 users) | Audio-based profanity detection |
| **Various from hatespeechdata.com**[^24] | 25+ languages (Spanish, Arabic, German, Italian, etc.) | Varies by language | Comprehensive catalog |


***

## Unlabeled Datasets

**Gaming Chat/Usernames/Guild Names:**

Unfortunately, **there are very few publicly available unlabeled gaming datasets** specifically for usernames and guild names. Most research focuses on in-game chat. However:

1. **Reddit Gaming Communities**: 300,000+ comments from gaming subreddits (r/VideoGames, r/HeartsOfIron) - available for research[^25]
2. **Minecraft Server Chat**: Used in leetspeak detection research - 1,000 chat samples mentioned[^26]
3. **Game-specific APIs**: Many games provide APIs that could be used to collect public data:
    - Steam Community profiles
    - League of Legends player names (Riot API)
    - Overwatch player profiles

**General Social Media** (useful for training baseline models):

- Twitter archives (requires academic access)
- Reddit comment dumps (available via academic torrents)
- YouTube comments (can be scraped with appropriate permissions)

***

## Python Implementations

### Recommended Libraries

**1. better-profanity**[^27]

- **Pros**: Fast, supports leetspeak variants, customizable wordlists, Unicode support
- **Speed**: Blazingly fast using string comparison vs regex
- **Installation**: `pip install better-profanity`
- **Features**:
    - Detects modified spellings (p0rn, h4NDjob)
    - Custom censor characters
    - Whitelist/blacklist support
    - Word and full-text censoring

```python
from better_profanity import profanity
profanity.load_censor_words()
text = "You p1ec3 of sHit."
censored = profanity.censor(text)  # "You **** of ****."
```

**2. profanity-check**[^28]

- **Pros**: ML-based (linear SVM), trained on 200k labeled samples, no explicit blacklist
- **Performance**: 300-4000x faster than profanity-filter
- **Accuracy**: 95% test accuracy, 93% balanced accuracy
- **Installation**: `pip install profanity-check` (or `alt-profanity-check` for maintained version)[^29]
- **Use case**: Better for detecting toxic intent vs simple word matching

```python
from profanity_check import predict
predict(['You cocksucker'])  # Returns [^1] (profane)
```

**3. profanity-filter**[^30]

- **Pros**: Deep analysis with Levenshtein automata, multilingual (English/Russian), Spacy integration
- **Cons**: Much slower than alternatives (60ms vs 0.2ms per prediction)
- **Features**:
    - Context-aware to some extent
    - Detects derivatives of profane words
    - Explanation of decisions
    - RESTful web service

**4. Transformer-based Models** (for more sophisticated detection)

- **ToxicBERT**: Fine-tuned BERT for toxicity[^26]
- **HateBERT**: Specialized for hate speech
- **Better for**: Context understanding, implicit toxicity, gaming-specific context


### Comparison Matrix

| Library | Speed | Accuracy | Context-Aware | Leetspeak | Multilingual | Best For |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| better-profanity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ✅ | Limited | Fast filtering, usernames |
| profanity-check | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | English only | General toxicity detection |
| profanity-filter | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ✅ | EN/RU | Deep analysis needed |
| ToxicBERT/ML | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Varies | Gaming chat with context |


***

<span style="display:none">[^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/pdf/2205.01374.pdf

[^2]: https://www.sciencedirect.com/science/article/abs/pii/S0952197624003178

[^3]: https://uu.diva-portal.org/smash/get/diva2:1437440/FULLTEXT01.pdf

[^4]: https://cleanspeak.com/blog-archive/2013/06/06/how-to-filter-and-moderate-usernames

[^5]: https://www.reddit.com/r/gamedesign/comments/ojcwfi/banned_usernames_dataset/

[^6]: https://www.nbcnews.com/tech/video-games/online-games-struggle-rein-hateful-usernames-report-finds-rcna95605

[^7]: http://arxiv.org/pdf/2211.05995.pdf

[^8]: https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain

[^9]: https://hatespeech.berkeley.edu

[^10]: https://lmsys.org/blog/2023-10-30-toxicchat/

[^11]: https://arxiv.org/pdf/2311.06446.pdf

[^12]: http://arxiv.org/pdf/2403.11456.pdf

[^13]: https://eecs.ku.edu/battling-toxicity-comparative-analysis-machine-learning-models-content-moderation

[^14]: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

[^15]: https://github.com/surge-ai/toxicity

[^16]: https://aclanthology.org/2021.findings-acl.213.pdf

[^17]: https://arxiv.org/pdf/2106.06213.pdf

[^18]: https://aclanthology.org/2025.naacl-short.37.pdf

[^19]: https://arxiv.org/html/2510.17924v1

[^20]: https://arxiv.org/pdf/2106.01598.pdf

[^21]: https://aclanthology.org/2023.findings-emnlp.663.pdf

[^22]: http://arxiv.org/pdf/2310.18330.pdf

[^23]: https://arxiv.org/abs/2202.07991

[^24]: https://hatespeechdata.com

[^25]: https://gnet-research.org/2022/01/20/exploring-extreme-language-in-gaming-communities/

[^26]: https://technoaretepublication.org/computer-applications/article/enhancement-of-profanity.pdf

[^27]: https://pypi.org/project/better-profanity/

[^28]: https://github.com/vzhou842/profanity-check

[^29]: https://pypi.org/project/alt-profanity-check/

[^30]: https://pypi.org/project/profanity-filter/

[^31]: https://aclanthology.org/2021.acl-long.132.pdf

[^32]: http://arxiv.org/pdf/2411.19832.pdf

[^33]: https://arxiv.org/pdf/2409.14740.pdf

[^34]: https://arxiv.org/pdf/2402.03221.pdf

[^35]: http://arxiv.org/pdf/2305.14081.pdf

[^36]: https://purplescape.com/profanity-in-llm/

[^37]: https://arxiv.org/html/2511.11599v1

[^38]: https://github.com/aymeam/Datasets-for-Hate-Speech-Detection

[^39]: https://www.kaggle.com/datasets/atharvasoundankar/gen-ai-misinformation-detection-datase-20242025

[^40]: https://cleanlab.ai/blog/learn/text-content-moderation/

[^41]: https://safetyprompts.com

[^42]: https://cacm.acm.org/blogcacm/the-ugc-overload-scaling-content-moderation-for-massive-datasets/

[^43]: https://spj.science.org/doi/10.34133/research.0189

[^44]: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset

[^45]: https://www.promptfoo.dev/blog/top-llm-safety-bias-benchmarks/

[^46]: https://www.sciencedirect.com/science/article/pii/S2352340922010356

[^47]: https://www.ijfmr.com/papers/2024/2/14927.pdf

[^48]: https://arxiv.org/pdf/2311.14685.pdf

[^49]: https://toknowpress.net/ISBN/978-961-6914-20-8/117.pdf

[^50]: https://atlarge-research.com/pdfs/2015-martens-toxicity.pdf

[^51]: https://stackoverflow.com/questions/273516/how-do-you-implement-a-good-profanity-filter

[^52]: https://areebbeigh.github.io/profanityfilter/

[^53]: https://www.reddit.com/r/startups/comments/iucsua/profanity_filters_for_usernames/

[^54]: https://towardsdatascience.com/build-your-language-filter-with-python-d6502f9c224b/

[^55]: https://ntrs.nasa.gov/citations/20220001514

[^56]: https://greip.io/blog/Profanity-Detection-How-AI-Can-Help-Maintain-Online-Etiquette-34

[^57]: https://stackoverflow.com/questions/3531746/what-s-a-good-python-profanity-filter-library

[^58]: https://huggingface.co/datasets/lmsys/toxic-chat

[^59]: https://www.spectrumlabsai.com/profanity-filters

[^60]: https://www.geeksforgeeks.org/python/censor-bad-words-in-python-using-better-profanity/

[^61]: https://www.reddit.com/r/leagueoflegends/comments/llp4q1/trained_a_neural_net_to_recognize_toxic_lol/

[^62]: https://chekkee.com/6-common-tricks-used-to-avoid-profanity-filters/

[^63]: https://arxiv.org/pdf/2311.03449.pdf

[^64]: https://www.aclweb.org/anthology/2021.naacl-main.182.pdf

[^65]: https://arxiv.org/pdf/2403.18957.pdf

[^66]: https://us.forums.blizzard.com/en/wow/t/offensive-names-guild/1883112

[^67]: https://dev.to/frankdev20/how-to-use-the-python-betterprofanity-filter-api-with-graphql-43if

[^68]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11042037/

[^69]: https://www.clcoding.com/2024/04/python-library-to-filter-and-censor.html

[^70]: https://www.basedlabs.ai/tools/offensive-gamertags

[^71]: https://www.ums.ac.id/en/news/research/multilingual-and-multidomain-environment-posing-more-challenge-for-artificial-intelligence-hate-speech-detection

[^72]: https://pypi.org/project/better-profanity/0.1/

[^73]: https://www.redguides.com/community/threads/ridiculous-funny-weird-guild-names.85499/

[^74]: https://www.youtube.com/watch?v=Refyih_dRt0

[^75]: https://en-forum.guildwars2.com/topic/89082-inappropriate-guild-names-need-a-report-button/

