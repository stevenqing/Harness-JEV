SCIENCEWORLD_ACTION_SCHEMA = """- open/close OBJ
- activate/deactivate OBJ
- connect OBJ to OBJ
- disconnect OBJ
- use OBJ [on OBJ]
- look around
- look at OBJ
- look in OBJ
- read OBJ
- move OBJ to OBJ
- pick up OBJ
- put down OBJ
- pour OBJ into OBJ
- dunk OBJ into OBJ
- mix OBJ
- go to LOC
- eat OBJ
- flush OBJ
- focus on OBJ
- wait1"""

SCIENCEWORLD_TEMPLATE_NO_HIS = """You are controlling ScienceWorld. First write one brief Thought, then write exactly one environment command. Do not write any other text.

Required format:
Thought: <one concise sentence>
Action: <one ScienceWorld command>

Task:
{task_description}

Available action formats:
{action_schema}

Initial observation:
{current_observation}
"""

SCIENCEWORLD_TEMPLATE = """You are controlling ScienceWorld. First write one brief Thought, then write exactly one environment command. Do not write any other text.

Required format:
Thought: <one concise sentence>
Action: <one ScienceWorld command>

Task:
{task_description}

Available action formats:
{action_schema}

Initial observation:
{initial_observation}

Recent interaction history:
{action_history}
"""

SCIENCEWORLD_TEMPLATE_CURRENT = """You are controlling ScienceWorld. First write one brief Thought, then write exactly one environment command. Do not write any other text.

Required format:
Thought: <one concise sentence>
Action: <one ScienceWorld command>

Task:
{task_description}

Available action formats:
{action_schema}

Current observation:
{current_observation}
"""
