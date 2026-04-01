memory-eval run \
    --task healthbench \
    --model-config azure-gpt-4o

memory-eval evaluate \
    --task healthbench \
    --model-config azure-gpt-4o \
    --grader-config qwen3.5-122b-a10b