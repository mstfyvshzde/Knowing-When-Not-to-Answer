# High-Entailment Incorrect Predictions

These cases were automatically selected because the QA prediction was incorrect while the question-aware verifier assigned high entailment. Error categories must be assigned manually.

## Annotation categories

- `RETRIEVAL_ERROR`
- `QA_PREDICTION_ERROR`
- `CLAIM_GENERATION_ERROR`
- `NLI_ERROR`
- `ANNOTATION_AMBIGUITY`
- `EVIDENCE_SUPPORTS_WRONG_ANSWER`
- `MULTIPLE_FAILURES`
- `OTHER`

## Case 1

| Field | Value |
|---|---|
| Record index | 64 |
| Question | What already existed in the USSR before 1973? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | a negotiated settlement based on equality between both parties |
| Confidence | 0.039728 |
| Entailment probability | 0.990564 |
| Contradiction probability | 0.001406 |
| Neutral probability | 0.008030 |
| NLI label | ENTAILMENT |
| Claim | a negotiated settlement based on equality between both parties already existed in the ussr before 1973. |
| Evidence | In the United States, scholars argue that there already existed a negotiated settlement based on equality between both parties prior to 1973. The possibility that the Middle East could become another superpower confrontation with the USSR was of |
| Retriever score | 0.930000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 2

| Field | Value |
|---|---|
| Record index | 38 |
| Question | What is the universal band that digital recievers will receive free to air channels on? |
| Gold answer(s) | Ku band \| 9.75/10.600 GHz \| universal Ku band |
| Answerable | true |
| Prediction | Ku band LNB |
| Confidence | 0.766258 |
| Entailment probability | 0.985532 |
| Contradiction probability | 0.005962 |
| Neutral probability | 0.008506 |
| NLI label | ENTAILMENT |
| Claim | the universal band that digital recievers will receive free to air channels on is ku band lnb. |
| Evidence | Provided is a universal Ku band LNB (9.75/10.600 GHz) which is fitted at the end of the dish and pointed at the correct satellite constellation; most digit |
| Retriever score | 0.727778 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 3

| Field | Value |
|---|---|
| Record index | 190 |
| Question | Who invaded Scotland in the 10th century? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | William |
| Confidence | 0.045821 |
| Entailment probability | 0.983715 |
| Contradiction probability | 0.007311 |
| Neutral probability | 0.008975 |
| NLI label | ENTAILMENT |
| Claim | william invaded scotland in the 10th century. |
| Evidence | One of the claimants of the English throne opposing William the Conqueror, Edgar Atheling, eventually fled to Scotland. King Malcolm III of Scotland married Edgar's sister Margare |
| Retriever score | 0.737500 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 4

| Field | Value |
|---|---|
| Record index | 19 |
| Question | What is a common occurrence during summer days? |
| Gold answer(s) | thunderstorms |
| Answerable | true |
| Prediction | thunderstorms to erupt |
| Confidence | 0.921626 |
| Entailment probability | 0.982008 |
| Contradiction probability | 0.000677 |
| Neutral probability | 0.017315 |
| NLI label | ENTAILMENT |
| Claim | thunderstorms to erupt is a common occurrence during summer days. |
| Evidence | °C) possible. The highest temperature recorded was 104 °F (40 °C) on July 11, 1879 and July 28, 1872. It is common for thunderstorms to erupt during a typical summer afternoon. These are caused by the rapid heating of the land relative to the water, combined wi |
| Retriever score | 0.860000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 5

| Field | Value |
|---|---|
| Record index | 66 |
| Question | What separates the neuroimmune system and peripheral immune system in humans? |
| Gold answer(s) | blood–brain barrier, blood–cerebrospinal fluid barrier \| blood–brain barrier, blood–cerebrospinal fluid barrier, and similar fluid–brain barriers \| fluid–brain barriers |
| Answerable | true |
| Prediction | blood–brain barrier |
| Confidence | 0.748079 |
| Entailment probability | 0.980511 |
| Contradiction probability | 0.008714 |
| Neutral probability | 0.010775 |
| NLI label | ENTAILMENT |
| Claim | the blood–brain barrier separates the neuroimmune system and peripheral immune system in humans. |
| Evidence | nate immune system versus the adaptive immune system, or humoral immunity versus cell-mediated immunity. In humans, the blood–brain barrier, blood–cerebrospinal fluid barrier, and similar fluid–brain barriers separate the peripheral immune system from the neu |
| Retriever score | 0.883333 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 6

| Field | Value |
|---|---|
| Record index | 36 |
| Question | NSF developed into what through public funding and private partnerships? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | Internet backbone |
| Confidence | 0.505570 |
| Entailment probability | 0.979104 |
| Contradiction probability | 0.004862 |
| Neutral probability | 0.016034 |
| NLI label | ENTAILMENT |
| Claim | nsf developed into internet backbone through public funding and private partnerships. |
| Evidence | mputing centers, through further public funding and private industry partnerships it developed into a major part of the Internet backbone. |
| Retriever score | 0.950000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 7

| Field | Value |
|---|---|
| Record index | 105 |
| Question | What Major Soccer League teams played in Los Angeles in 2014? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | LA Galaxy and Chivas USA |
| Confidence | 0.782004 |
| Entailment probability | 0.975128 |
| Contradiction probability | 0.017281 |
| Neutral probability | 0.007591 |
| NLI label | ENTAILMENT |
| Claim | la galaxy and chivas usa played in los angeles in 2014. |
| Evidence | From 2005 to 2014, there were two Major League Soccer teams in Los Angeles — the LA Galaxy and Chivas USA — that both played at the StubHub Center and were local rivals. However, Chivas were suspended following the 2014 MLS s |
| Retriever score | 1.000000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 8

| Field | Value |
|---|---|
| Record index | 9 |
| Question | What is the cryogenic and chemical compound storage method? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | Oxygen |
| Confidence | 0.300576 |
| Entailment probability | 0.972650 |
| Contradiction probability | 0.002859 |
| Neutral probability | 0.024491 |
| NLI label | ENTAILMENT |
| Claim | the cryogenic and chemical compound storage method is oxygen. |
| Evidence | Oxygen storage methods include high pressure oxygen tanks, cryogenics and chemical compounds. For reasons of economy, oxygen i |
| Retriever score | 0.790000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 9

| Field | Value |
|---|---|
| Record index | 78 |
| Question | What did John Dalton hypothesize in 1805? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | all elements were monatomic |
| Confidence | 0.539738 |
| Entailment probability | 0.969111 |
| Contradiction probability | 0.016838 |
| Neutral probability | 0.014052 |
| NLI label | ENTAILMENT |
| Claim | john dalton hypothesized all elements were monatomic in 1805. |
| Evidence | John Dalton's original atomic hypothesis assumed that all elements were monatomic and that the atoms in compounds would normally have the simplest atomic ratios with respect to one another. For example |
| Retriever score | 0.737500 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 10

| Field | Value |
|---|---|
| Record index | 120 |
| Question | What type of microscope is used by petrologists? |
| Gold answer(s) | petrographic microscope \| petrographic |
| Answerable | true |
| Prediction | optical microscopy |
| Confidence | 0.556435 |
| Entailment probability | 0.967105 |
| Contradiction probability | 0.008032 |
| Neutral probability | 0.024864 |
| NLI label | ENTAILMENT |
| Claim | optical microscopy is used by petrologists. |
| Evidence | identify rock samples in the laboratory. Two of the primary methods for identifying rocks in the laboratory are through optical microscopy and by using an electron microprobe. In an optical mineralogy analysis, thin sections of rock samples are analyzed thro |
| Retriever score | 0.650000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 11

| Field | Value |
|---|---|
| Record index | 117 |
| Question | What was implemented with an X.25 interface? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | DATAPAC and TRANSPAC |
| Confidence | 0.360331 |
| Entailment probability | 0.966639 |
| Contradiction probability | 0.012495 |
| Neutral probability | 0.020866 |
| NLI label | ENTAILMENT |
| Claim | datapac and transpac was implemented with an x25 interface. |
| Evidence | There were two kinds of X.25 networks. Some such as DATAPAC and TRANSPAC were initially implemented with an X.25 external interface. Some older networks such as TELENET and TYMNET were modifie |
| Retriever score | 1.000000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 12

| Field | Value |
|---|---|
| Record index | 43 |
| Question | How much funding did the RAND Corporation receive? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | US Department of Defense |
| Confidence | 0.558731 |
| Entailment probability | 0.966285 |
| Contradiction probability | 0.005163 |
| Neutral probability | 0.028552 |
| NLI label | ENTAILMENT |
| Claim | the rand corporation received us department of defense funding. |
| Evidence | ient routing method for telecommunication messages as part of a research program at the RAND Corporation, funded by the US Department of Defense. This concept contrasted and contradicted the theretofore established principles of pre-allocation of network bandwidth |
| Retriever score | 0.790000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 13

| Field | Value |
|---|---|
| Record index | 124 |
| Question | How many professional schools does the University of Higher Learning consist of? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | seven |
| Confidence | 0.062095 |
| Entailment probability | 0.965293 |
| Contradiction probability | 0.012738 |
| Neutral probability | 0.021969 |
| NLI label | ENTAILMENT |
| Claim | the university of higher learning consists of seven professional schools. |
| Evidence | The academic bodies of the University of Chicago consist of the College, four divisions of graduate research and seven professional schools. The university also contains a library system, the University of Chicago Press, the University of |
| Retriever score | 0.850000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 14

| Field | Value |
|---|---|
| Record index | 15 |
| Question | Who distributes money from the annual trust fund? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | the WMO, by UNEP, and by IPCC Members |
| Confidence | 0.328548 |
| Entailment probability | 0.955638 |
| Contradiction probability | 0.008734 |
| Neutral probability | 0.035628 |
| NLI label | ENTAILMENT |
| Claim | the wmo by unep and by ipcc members distributes money from the annual trust fund. |
| Evidence | by the WMO, while UNEP meets the cost of the Depute Secretary. Annual cash contributions to the Trust Fund are made by the WMO, by UNEP, and by IPCC Members; the scale of payments is determined by the IPCC Panel, which is also responsible for considering and adopting by conse |
| Retriever score | 0.860000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 15

| Field | Value |
|---|---|
| Record index | 32 |
| Question | Along with sport and art, what is a type of talent scholarship? |
| Gold answer(s) | academic |
| Answerable | true |
| Prediction | academic scholarship |
| Confidence | 0.791924 |
| Entailment probability | 0.954802 |
| Contradiction probability | 0.001008 |
| Neutral probability | 0.044190 |
| NLI label | ENTAILMENT |
| Claim | along with sport and art a type of talent scholarship is academic scholarship. |
| Evidence | hip, which makes the cost cheaper, depending on a talent the student may have (e.g. sport scholarship, art scholarship, academic scholarship), financial need, or tax credit scholarships that might be available. |
| Retriever score | 0.883333 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 16

| Field | Value |
|---|---|
| Record index | 65 |
| Question | What is stainless steel's theoretical Carnot efficiency? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | 63% |
| Confidence | 0.866735 |
| Entailment probability | 0.950218 |
| Contradiction probability | 0.008127 |
| Neutral probability | 0.041655 |
| NLI label | ENTAILMENT |
| Claim | stainless steels theoretical carnot efficiency is 63. |
| Evidence | it of stainless steel) and condenser temperatures are around 30 °C. This gives a theoretical Carnot efficiency of about 63% compared with an actual efficiency of 42% for a modern coal-fired power station. This low turbine entry temperature (co |
| Retriever score | 0.930000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 17

| Field | Value |
|---|---|
| Record index | 156 |
| Question | What did Marchall Cohen note about crime? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | civil disobedience |
| Confidence | 0.048381 |
| Entailment probability | 0.934257 |
| Contradiction probability | 0.007239 |
| Neutral probability | 0.058504 |
| NLI label | ENTAILMENT |
| Claim | marchall cohen noted civil disobedience about crime. |
| Evidence | It has been argued that the term "civil disobedience" has always suffered from ambiguity and in modern times, become utterly debased. Marshall Cohen notes, "It has been use |
| Retriever score | 0.737500 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 18

| Field | Value |
|---|---|
| Record index | 82 |
| Question | What, along with solar, coal, and nuclear, uses the heat process? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | biomass |
| Confidence | 0.495051 |
| Entailment probability | 0.921628 |
| Contradiction probability | 0.001999 |
| Neutral probability | 0.076373 |
| NLI label | ENTAILMENT |
| Claim | biomass along with solar coal and nuclear uses the heat process. |
| Evidence | ankine steam cycles generated about 90% of all electric power used throughout the world, including virtually all solar, biomass, coal and nuclear power plants. It is named after William John Macquorn Rankine, a Scottish polymath. |
| Retriever score | 0.800000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 19

| Field | Value |
|---|---|
| Record index | 29 |
| Question | What type of experiment did Philo of Pneumatica preform? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | inverting a vessel over a burning candle |
| Confidence | 0.593045 |
| Entailment probability | 0.917844 |
| Contradiction probability | 0.045832 |
| Neutral probability | 0.036323 |
| NLI label | ENTAILMENT |
| Claim | philo of pneumatica preformed inverting a vessel over a burning candle. |
| Evidence | ucted by the 2nd century BCE Greek writer on mechanics, Philo of Byzantium. In his work Pneumatica, Philo observed that inverting a vessel over a burning candle and surrounding the vessel's neck with water resulted in some water rising into the neck. Philo incorrectly surmised th |
| Retriever score | 0.790000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 20

| Field | Value |
|---|---|
| Record index | 121 |
| Question | What branch of theoretical computer class deals with broadly classifying computational problems by difficulty and class of relationship? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | Computational complexity theory |
| Confidence | 0.721664 |
| Entailment probability | 0.911712 |
| Contradiction probability | 0.034377 |
| Neutral probability | 0.053911 |
| NLI label | ENTAILMENT |
| Claim | computational complexity theory deals with broadly classifying computational problems by difficulty and class of relationship. |
| Evidence | Computational complexity theory is a branch of the theory of computation in theoretical computer science that focuses on classifying computational prob |
| Retriever score | 0.809091 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 21

| Field | Value |
|---|---|
| Record index | 181 |
| Question | What is the name of one algorithm useful for conveniently testing the primality of decimal digits? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | Miller–Rabin primality test |
| Confidence | 0.363256 |
| Entailment probability | 0.901940 |
| Contradiction probability | 0.020768 |
| Neutral probability | 0.077292 |
| NLI label | ENTAILMENT |
| Claim | the name of one algorithm useful for conveniently testing the primality of decimal digits is miller–rabin primality test. |
| Evidence | hms much more efficient than trial division have been devised to test the primality of large numbers. These include the Miller–Rabin primality test, which is fast but has a small probability of error, and the AKS primality test, which always produces the correct answ |
| Retriever score | 0.693750 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 22

| Field | Value |
|---|---|
| Record index | 165 |
| Question | What else were families with incomes below $38,000 not required to pay for in 2009? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | room and board |
| Confidence | 0.078771 |
| Entailment probability | 0.900472 |
| Contradiction probability | 0.022859 |
| Neutral probability | 0.076669 |
| NLI label | ENTAILMENT |
| Claim | families with incomes below 38000 were not required to pay for room and board in 2009. |
| Evidence | nce of $57,000. Beginning 2007, families with incomes below $60,000 pay nothing for their children to attend, including room and board. Families with incomes between $60,000 to $80,000 pay only a few thousand dollars per year, and families earning betwee |
| Retriever score | 0.805556 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 23

| Field | Value |
|---|---|
| Record index | 153 |
| Question | How was the efficiency of a concept engine typically evaluated? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | duty |
| Confidence | 0.604280 |
| Entailment probability | 0.894975 |
| Contradiction probability | 0.040294 |
| Neutral probability | 0.064731 |
| NLI label | ENTAILMENT |
| Claim | the efficiency of a concept engine was typically evaluated by duty. |
| Evidence | The historical measure of a steam engine's energy efficiency was its "duty". The concept of duty was first introduced by Watt in order to illustrate how much more efficient his engines were over |
| Retriever score | 0.790000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 24

| Field | Value |
|---|---|
| Record index | 83 |
| Question | How many years has Bronze Age agriculture gone on for? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | ~3000 yr BP |
| Confidence | 0.097990 |
| Entailment probability | 0.890698 |
| Contradiction probability | 0.023996 |
| Neutral probability | 0.085306 |
| NLI label | ENTAILMENT |
| Claim | bronze age agriculture has gone on for 3000 yr bp. |
| Evidence | Since ~3000 yr BP (= years Before Present), human impact is seen in the delta. As a result of increasing land clearance (Bronze Age agric |
| Retriever score | 0.800000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 25

| Field | Value |
|---|---|
| Record index | 13 |
| Question | What are three basic primary resources used to gauge complexity? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | time and storage |
| Confidence | 0.458568 |
| Entailment probability | 0.873668 |
| Contradiction probability | 0.001105 |
| Neutral probability | 0.125227 |
| NLI label | ENTAILMENT |
| Claim | time and storage are three basic primary resources used to gauge complexity. |
| Evidence | cal models of computation to study these problems and quantifying the amount of resources needed to solve them, such as time and storage. Other complexity measures are also used, such as the amount of communication (used in communication complexity), the n |
| Retriever score | 0.800000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 26

| Field | Value |
|---|---|
| Record index | 98 |
| Question | What are two ways lava tubes are added during deformation? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | depositionally and intrusively |
| Confidence | 0.018533 |
| Entailment probability | 0.858055 |
| Contradiction probability | 0.023937 |
| Neutral probability | 0.118007 |
| NLI label | ENTAILMENT |
| Claim | two ways lava tubes are added during deformation are depositionally and intrusively. |
| Evidence | The addition of new rock units, both depositionally and intrusively, often occurs during deformation. Faulting and other deformational processes result in the creation of topographic grad |
| Retriever score | 0.750000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 27

| Field | Value |
|---|---|
| Record index | 14 |
| Question | What is the name of an integer in which addition, subtraction, and multiplication are defined? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | prime elements |
| Confidence | 0.537637 |
| Entailment probability | 0.843457 |
| Contradiction probability | 0.102479 |
| Neutral probability | 0.054064 |
| NLI label | ENTAILMENT |
| Claim | the name of an integer in which addition subtraction and multiplication are defined is prime elements. |
| Evidence | elements of any commutative ring R, an algebraic structure where addition, subtraction and multiplication are defined: prime elements and irreducible elements. An element p of R is called prime element if it is neither zero nor a unit (i.e., does not ha |
| Retriever score | 0.930000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |

## Case 28

| Field | Value |
|---|---|
| Record index | 187 |
| Question | How quickly can fish larvae grow? |
| Gold answer(s) | [UNANSWERABLE] |
| Answerable | false |
| Prediction | breed very rapidly |
| Confidence | 0.008169 |
| Entailment probability | 0.826986 |
| Contradiction probability | 0.087905 |
| Neutral probability | 0.085110 |
| NLI label | ENTAILMENT |
| Claim | fish larvae can breed very rapidly. |
| Evidence | ade new territories (although this was not predicted until after it so successfully colonized the Black Sea), as it can breed very rapidly and tolerate a wide range of water temperatures and salinities. The impact was increased by chronic overfishing, and by |
| Retriever score | 0.720000 |
| Evidence source |  |
| Claim valid | true |
| Invalid reasons |  |
| Primary error category |  |
| Secondary error category |  |
| Notes |  |
