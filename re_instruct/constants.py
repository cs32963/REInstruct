"""constants"""

VICUNA_V1_1_SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
REVERSE_PROMPT = "Below is a reponse from an AI Assistant and its user instruction. The instruction is used as prompt for the response."
REWRITE_PROMPT = "Answer the question based on the web text provided. Your answer must be helpful, detailed, and polite. Hide the fact that your answer is actually based on the web text, i.e. answer directly."

USER_HEAD = "\n### User:\n"
ASSISTANT_HEAD = "\n### Assistant:\n"
WEB_TEXT_HEAD = "\n### Web text:\n"
QUESTION_HEAD = "\n### Question:\n"
ANSWER_HEAD = "\n### Answer:\n"

SEED_PROMPT = "Answer in the style of an AI Assistant."
AUG_PROMPT = "Answer with knowledge from web search."
