# RAG vs fine-tuning — local knowledge note (generic AI knowledge, not Teja's work)

RAG (Retrieval-Augmented Generation) keeps a base LLM unchanged and gives it
retrieved passages at prompt time. It is the right choice when the knowledge
changes often, when answers must cite sources, or when you have documents but
little labelled training data. RAG is cheaper to update: add or replace a
document and the next answer can use it, with no retraining.

Fine-tuning trains the model weights on labelled examples. It is the right
choice when you need a consistent style, a specific output format, or behaviour
the base model does not have — for example a classification task or a narrow
domain tone. Fine-tuning does not reliably teach the model new facts, and it
cannot cite sources on its own. Fine-tuned knowledge also goes stale and needs
retraining to update.

For a doc chatbot, RAG is usually the better first choice: it grounds answers
in the documents, provides citations, and is easy to update. Fine-tuning can
be added later for style or format, and the two can be combined: fine-tune for
how to answer, retrieve for what to answer.
