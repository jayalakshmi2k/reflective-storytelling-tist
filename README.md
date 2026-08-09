Supplementary Material for TIST Reflective Storytelling Paper
-------------------------------------------------------------

This repository contains supplementary material for the ACM TIST manuscript on a person-tailored reflective storytelling agent for older adults.

The material is provided to support reproducibility and inspection of the study methods, narrative-generation conditions, grounding analysis, and argument-mining evaluation.


Repository contents
-------------------

annotation guidelines
---------------------

Contains the annotation guidelines used for the human annotation of argumentative discourse units (ADUs).

The annotation labels are:

- CLAIM
- PREMISE
- NONE

The guidelines also describe how ambiguous and mixed units were handled.


argument_mining
---------------

Contains files related to the argument-mining evaluation.

This includes human consensus annotations, model predictions, and error-analysis materials for Conditions A and B.

Argument mining was evaluated against the human consensus annotations.


data
----

Contains the data used for the technical analyses reported in the manuscript.

The 'generated_narratives' folder contains narratives from the three generation conditions:

- 'condition_A': full system with structured persona/activity information, grounding guidance, dialogue-purpose guidance, and argumentation-scheme guidance.
- 'condition_B': same architecture as Condition A, but without argumentation-scheme guidance.
- 'condition_C': standard LLM baseline without the structured person-tailored storytelling architecture.

The 'gold_annotations' folder contains the human consensus annotations used for the argument-mining evaluation.

The 'grounding' folder contains the data used for the persona-grounding analysis.


personas
--------

Contains the five predefined fictional personas used in the study.

The personas contain structured information about activities and were used to generate person-tailored narratives.

Participants in Phase II selected from these predefined personas. The narratives were therefore not generated from the participants' own personal activity data.


prompts
-------

Contains the prompts used for the narrative-generation conditions and argument-mining evaluation.

The generation prompts document the differences between Conditions A, B, and C.


reproducibility
---------------

Contains the scripts used to reproduce the technical analyses reported in the manuscript.

These include scripts for:

- argument-mining evaluation
- grounding analysis
- statistical analysis

The scripts use the corresponding files in the 'data' and 'argument_mining' folders.


study_materials
---------------

Contains materials used in the human evaluation, including the Phase II questionnaire materials and documentation of questionnaire variables.

The questionnaire material shows the questions and response options presented to participants.


system_code
-----------

Contains relevant implementation files from the storytelling system.

In the implementation, the storytelling agent may be named 'coach_agent'. This corresponds to the storytelling agent described in the manuscript.


Experimental conditions
-----------------------

The technical evaluation uses three generation conditions.

| Condition | Persona/activity information | Grounding guidance | Dialogue-purpose guidance | Argumentation-scheme guidance |
|-----------|------------------------------|--------------------|---------------------------|-------------------------------|
| A | Yes | Yes | Yes | Yes |
| B | Yes | Yes | Yes | No |
| C | No | No | No | No |

Condition A represents the full storytelling architecture.

Condition B is an ablation condition used to examine the incremental contribution of argumentation-scheme guidance.

Condition C is a standard LLM generation baseline.

Conditions A and B are used to examine the incremental effect of argumentation-scheme guidance.

Conditions A and C are used for the standard-LLM comparison.


Argument-mining datasets
------------------------

The argument-mining evaluation uses human consensus annotations for Conditions A and B.

- Dataset A: 529 ADUs
- Dataset B: 436 ADUs
- Total: 965 ADUs

Each ADU is labelled as CLAIM, PREMISE, or NONE.

Condition C was not included in the argument-mining gold-standard evaluation.


Grounding analysis
------------------

Grounding was assessed relative to the structured persona/activity information available to the system.

The grounding labels are:

- GROUNDED
- UNSUPPORTED
- CONTRADICTED
- NOT APPLICABLE

GROUNDED indicates that the statement is supported by the corresponding persona information.

UNSUPPORTED indicates that the statement introduces person-specific content that is not supported by the available persona information.

CONTRADICTED indicates that the statement conflicts with the available persona information.

NOT APPLICABLE indicates that the statement contains no person-specific content.

These labels concern consistency with the available persona information and should not be interpreted as general factual correctness or broad AI alignment.


Reproducing the analyses
------------------------

The scripts in the 'reproducibility' folder can be used with the corresponding files in 'data' and 'argument_mining'.

Outputs from the analyses are stored in the corresponding subfolders under 'results'.

The repository focuses on the material required to inspect and reproduce the analyses reported in the manuscript. Intermediate development files that are not required for reproducing the reported results are not included.


Notes on interpretation
-----------------------

The grounding analysis and argument-mining analysis address different aspects of the generated narratives.

Grounding concerns whether narrative content is supported by the available structured persona/activity information.

Argument mining concerns the identification of CLAIM, PREMISE, and NONE units in the narratives.

Argument-mining predictions should therefore not be interpreted as proof of correct reasoning, factual correctness, or grounding.


Study scope
-----------

Phase I was a formative evaluation with domain experts.

Phase II evaluated narratives with older adults using predefined fictional personas.

The technical evaluation (phase III) examines grounding, the effect of argumentation-scheme guidance, comparison with a standard LLM baseline, and the performance and limitations of the argument-mining inspection layer.


Manuscript
----------

Supplementary material for the ACM Transactions on Intelligent Systems and Technology (TIST) submission:

TIST-2026-02-0170